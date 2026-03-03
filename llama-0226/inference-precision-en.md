# llama.cpp Inference Precision Analysis (CPU / CUDA / Metal)

This document analyzes the precision behavior of `ggml_mul_mat()` — ggml's matrix multiplication operator and the core computation behind all Transformer linear layers — during quantized model inference in llama.cpp.

## 1. Quantized Matmul: The Complete Pipeline

### 1.1 CPU / CUDA: Dynamic WxA8

On CPU and CUDA, quantized matmul **temporarily quantizes FP32 activations to Q8 (INT8) online**, so that both sides become integers and INT8×INT8 hardware instructions can be used:

```
Weights (Q4_K, 4-bit storage)            Activations (FP32)
        |                                      |
   Unpack 4-bit -> INT8                quantize_row_q8_K()
   (pure bit ops, scale not applied)    FP32 -> INT8 online quantization
        |                                      |
        v                                      v
   INT8 (weight side)                     INT8 (activation side)
        |                                      |
        +---------- INT8 x INT8 --------------+
                        |
                   INT32 accumulation
                        |
                x FP32 scale (Q4_K.d x Q8_K.d)
                        |
                   FP32 output
```

The weight-side 4-bit -> INT8 is **unpacking** (pure bit operations, values unchanged), not dequantization. The activation-side FP32 -> INT8 is real quantization (with precision loss).

### 1.2 Metal: Dequantize to FP32

Metal does not use the INT8 path. It **dequantizes weights to FP32** (applies scale to recover floating-point values), while activations remain FP32 as-is:

```
Weights (Q4_K, 4-bit storage)            Activations (FP32)
        |                                      |
  dequantize_q4_K()                       Kept as-is
  Dequantize to FP32 (apply scale)             |
        |                                      |
        v                                      v
   FP32 (weight side)                     FP32 (activation side)
        |                                      |
        +---------- FP32 x FP32 --------------+
                        |
                   FP32 accumulation
                        |
                   FP32 output
```

Reason: Apple GPUs lack efficient INT8 matmul instructions; FP32 compute units are faster. Quantization on Metal only saves memory bandwidth, not compute.

### 1.3 Precision Impact

| | CPU / CUDA (INT8 path) | Metal (FP32 path) |
|--|----------------------|------------------|
| Weight precision loss | Q4_K quantization (same) | Q4_K quantization (same) |
| Activation precision loss | FP32 -> Q8 **extra quantization loss** | **None** (stays FP32) |
| Total quantization loss sources | **Two** (weights + activations) | **One** (weights only) |

CPU/CUDA trades one extra activation quantization loss for the speed advantage of INT8 integer instructions. Metal has higher precision but lower compute efficiency. This is a **speed vs. precision tradeoff**.

## 2. Online Activation Quantization: Implementation Details

### 2.1 CPU

Source: `ggml/src/ggml-cpu/ggml-cpu.c` — `ggml_compute_forward_mul_mat()`

```c
// Line 1241: Look up the paired Q8 type based on weight type
// Q4_K -> vec_dot_type = Q8_K (hardcoded in type_traits_cpu table)
enum ggml_type const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;

// Lines 1291-1326: FP32 activations -> Q8 quantization (unconditional)
if (src1->type != vec_dot_type) {           // FP32 != Q8_K -> always enters
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    from_float(src1->data, wdata, ne10);    // quantize_row_q8_K()
}
```

**There is no branch choosing "INT8 path or FP32 path" — the code structure itself has only this one path.**

### 2.2 CUDA

Source: `ggml/src/ggml-cuda/mmq.cu`, `quantize.cu`

```cuda
// mmq.cu line 73: Assert activations are FP32
GGML_ASSERT(src1->type == GGML_TYPE_F32);

// mmq.cu lines 122-137: Allocate buffer on GPU, quantize FP32 activations to Q8_1
ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), ...);
```

GPU quantization kernel (`quantize.cu` lines 176-271):

```cuda
const float4 xi = x4[...];                     // Load 4 FP32 values
float amax = max(abs(xi.x), abs(xi.y), ...);   // Find max absolute value
amax = warp_reduce_max(amax);                   // Warp-level reduction
const float d_inv = 127.0f / amax;              // Compute scale
char4 q;
q.x = roundf(xi.x * d_inv);                    // FP32 -> INT8
q.y = roundf(xi.y * d_inv);
q.z = roundf(xi.z * d_inv);
q.w = roundf(xi.w * d_inv);
yqs4[iqs/4] = q;                               // Write back 4 int8 values
```

### 2.3 Q8 Activation Quantization Function (CPU Reference)

Source: `ggml/src/ggml-quants.c` — `quantize_row_q8_K_ref()`

```c
void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k) {
    for (int i = 0; i < nb; i++) {
        // 1. Find max absolute value in block (256 values)
        float amax = 0;
        for (int j = 0; j < QK_K; ++j)
            amax = MAX(amax, fabsf(x[j]));

        // 2. Compute scale: map FP32 -> [-127, 127]
        const float iscale = -127.f / max;

        // 3. Quantize each FP32 -> INT8
        for (int j = 0; j < QK_K; ++j)
            y[i].qs[j] = MIN(127, nearest_int(iscale * x[j]));

        // 4. Store scale
        y[i].d = 1 / iscale;
    }
}
```

### 2.4 Weight Type to Q8 Variant Pairing

Defined in `type_traits_cpu[]` (`ggml-cpu.c` lines 207-390):

| Weight Type | Paired Q8 Type | vec_dot Function |
|------------|---------------|-----------------|
| Q4_0 | Q8_0 (block 32) | `ggml_vec_dot_q4_0_q8_0` |
| Q4_1 | Q8_1 (block 32) | `ggml_vec_dot_q4_1_q8_1` |
| Q5_0 | Q8_0 (block 32) | `ggml_vec_dot_q5_0_q8_0` |
| Q5_1 | Q8_1 (block 32) | `ggml_vec_dot_q5_1_q8_1` |
| Q2_K | Q8_K (block 256) | `ggml_vec_dot_q2_K_q8_K` |
| Q3_K | Q8_K (block 256) | `ggml_vec_dot_q3_K_q8_K` |
| Q4_K | Q8_K (block 256) | `ggml_vec_dot_q4_K_q8_K` |
| Q5_K | Q8_K (block 256) | `ggml_vec_dot_q5_K_q8_K` |
| Q6_K | Q8_K (block 256) | `ggml_vec_dot_q6_K_q8_K` |

There are three Q8 variants:

| Variant | Block Size | Scale Precision | Paired With |
|---------|-----------|----------------|------------|
| Q8_0 | 32 | FP16 | Q4_0 / Q5_0 |
| Q8_1 | 32 | FP32 | Q4_1 / Q5_1 |
| Q8_K | 256 | FP32 | All K-quants |

**Q8's primary role is not weight quantization — it is the temporary online quantization format for activations.**

### 2.5 Comparison with Traditional W8A8

llama.cpp's approach is essentially **dynamic WxA8**, but with key differences from traditional W8A8:

| | Traditional W8A8 | llama.cpp |
|--|-----------------|-----------|
| Activation quantization timing | Offline/static (scales determined by calibration dataset) | **Online/dynamic** (computed in real-time before each matmul) |
| Scale granularity | Per-tensor or per-channel | **Per-block** (every 32 or 256 values) |
| Inter-layer propagation | INT8 propagated between layers, errors **accumulate across layers** | Returns to FP32 after each layer, errors **do not accumulate** |

```
Traditional W8A8:
  -> [INT8] -> Layer1 -> [INT8] -> Layer2 -> [INT8] ->    Errors accumulate across layers

llama.cpp:
  -> [FP32] -> Layer1 -> [FP32] -> Layer2 -> [FP32] ->    Errors do NOT accumulate
                 | temp Q8      | temp Q8
              back to FP32   back to FP32
```

llama.cpp achieves higher precision (dynamic scales + fine granularity + no cross-layer accumulation), at the cost of an extra quantization operation per layer.

### 2.6 Scope of Q8 Quantization: Regular Activations, Not Just KV Cache

A common misconception is that "activation quantization only affects KV Cache." In reality, Q8 quantization affects **the hidden state of every linear layer that uses quantized weights**.

Each Transformer layer has two categories of `ggml_mul_mat`. Only the first triggers Q8 activation quantization:

**Category 1: Weight Projections (triggers Q8 quantization)**

```
src0 = quantized weights (Q4_K)    src1 = FP32 hidden state (regular activation)

build_lora_mm(wq, cur)       → ggml_mul_mat(wq, cur)       // Q projection
build_lora_mm(wk, cur)       → ggml_mul_mat(wk, cur)       // K projection
build_lora_mm(wv, cur)       → ggml_mul_mat(wv, cur)       // V projection
build_lora_mm(wo, attn_out)  → ggml_mul_mat(wo, attn_out)  // Output projection
build_lora_mm(ffn_up, cur)   → ggml_mul_mat(ffn_up, cur)   // FFN up
build_lora_mm(ffn_gate, cur) → ggml_mul_mat(ffn_gate, cur) // FFN gate
build_lora_mm(ffn_down, cur) → ggml_mul_mat(ffn_down, cur) // FFN down
```

Source: `src/models/llama.cpp` lines 46-116

`src0` (weights) is a quantized type → `type_traits_cpu[src0->type].vec_dot_type` table lookup yields Q8_K → `src1` (hidden state) is quantized to Q8. **This happens 7 times per layer (once for each linear layer).**

**Category 2: Attention K×Q / V×KQ (not triggered by default)**

```
src0 = K or V (from KV Cache, default F16)    src1 = Q or KQ (FP32/F16)

ggml_mul_mat(k, q)     // K×Q attention score    (llama-graph.cpp line 1798)
ggml_mul_mat(v, kq)    // V×KQ attention output  (llama-graph.cpp line 1842)
```

KV Cache default type is **F16** (`llama-context.cpp` lines 2792-2793), not a quantized format. Since `src0` is not a quantized type → the `type_traits_cpu` quantization table is not consulted → **Q8 activation quantization is not triggered**, and the FP path is used instead.

> Users can set `--cache-type-k q8_0` / `--cache-type-v q8_0` to store KV Cache in quantized format, which would trigger activation quantization for Attention matmuls as well. However, this is an explicit user choice, not the default behavior.

**Summary:**

| Matmul Type | src0 | src1 | Q8 Quantization Triggered? |
|---|---|---|---|
| Weight projections (QKV/FFN, 7 per layer) | Q4_K (quantized weights) | FP32 hidden state | **Yes, every time** |
| Attention K×Q | F16 (KV Cache, default) | F16/FP32 (Q vectors) | **No** |
| Attention V×KQ | F16 (KV Cache, default) | FP32 (attention weights) | **No** |

## 3. Matmul Compute Precision Across Three Backends

### 3.1 CPU

Source: `ggml/src/ggml-cpu/quants.c`, `arch/x86/quants.c`, `arch/arm/quants.c`

**Quantized matmul has only one path: integer arithmetic.** There is no FP32 fallback — there is no branch that says "if the CPU lacks INT8 instructions, fall back to FP32 matmul."

**Scalar fallback (CPUs without SIMD):** Any CPU can execute this path, as it only uses basic C integer operations:

```c
// Q4_0 x Q8_0 scalar path (quants.c lines 137-141)
const int v0 = (x[ib].qs[j] & 0x0F) - 8;   // Unpack to int (value range [-8, 7])
sumi0 += (v0 * y[ib].qs[j]);                // int x int8_t -> int

// Q4_K x Q8_K scalar path (quants.c lines 603-604)
aux16[l] = q8[l] * a[l];                    // int8_t x int8_t -> int16_t
aux32[l] += scale * aux16[l];               // int32_t accumulation
```

Note: In the scalar path, the **value range** is INT8 (-128 to 127), but the **C types** are not all `int8_t` — the Q4_0 path stores unpacked values as `int` (typically 32-bit), so the multiplication is `int × int8_t`. This does not affect correctness since the value range is unchanged.

**SIMD-accelerated paths:** VNNI, DOTPROD, etc. are not required — they only affect **parallelism and speed**, not the computation logic:

| Instruction Set | Multiply Precision | Accumulate Precision |
|----------------|-------------------|---------------------|
| No SIMD (scalar fallback) | int x int8_t -> int | int accumulation |
| SSSE3 `_mm_maddubs_epi16` | uint8 x int8 -> INT16 | INT16 -> INT32 |
| AVX512-VNNI `_mm256_dpbusd_epi32` | uint8 x int8 -> INT32 | INT32 |
| AMX-INT8 | INT8 x INT8 -> INT32 | INT32 (matrix-level) |
| NEON+DOTPROD `vdotq_s32` | int8 x int8 -> INT32 | INT32 |
| NEON (no DOTPROD) `vmull_s8` | int8 x int8 -> INT16 | INT16 -> INT32 |
| SVE `svdot_s32` | int8 x int8 -> INT32 | INT32 |

SIMD paths are selected at compile time via `__AVX512VNNI__`, `__ARM_FEATURE_DOTPROD__`, etc. Not runtime dispatch.

Non-quantized matmul: FP32×FP32 -> FP32.

### 3.2 CUDA

Source: `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmq.cu`, `vecdotq.cuh`, `quantize.cu`

#### Quantized Matmul: Automatic MMQ vs cuBLAS Selection

Decided by `ggml_cuda_should_use_mmq()` (`mmq.cu` lines 262-366):

```
GPU does not support dp4a (compute capability < 6.1)
  -> cuBLAS (dequantize -> FP16/FP32)

GPU supports dp4a but lacks FP16 Tensor Cores
  -> MMQ (INT8xINT8)

GPU has both dp4a and FP16 Tensor Cores (RTX 20xx/30xx/40xx)
  -> batch size < MMQ_DP4A_MAX_BATCH_SIZE -> MMQ (INT8)
  -> large batch size -> cuBLAS (dequantize -> FP16, higher Tensor Core throughput)
```

Compile-time overrides available: `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS`.

#### MMQ Path

```cuda
int sumi = 0;
sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);   // 4x int8xint8 -> INT32
return d4 * (sumi * ds8f.x - ...);              // FP32 scaling
```

#### cuBLAS Path (non-quantized / fallback precision)

| Input Type | GPU Condition | Multiply Precision | Accumulate Precision |
|-----------|--------------|-------------------|---------------------|
| FP16 | CDNA / RDNA4 | FP16 | **FP32** |
| FP16 | Other consumer GPUs + `GGML_PREC_DEFAULT` | **FP16** | **FP16** |
| FP16 | Any + `GGML_PREC_F32` | FP16 | **FP32** |
| BF16 | Any | BF16 | **FP32** |
| FP32 | Any | FP32 | FP32 |

### 3.3 Metal

Source: `ggml/src/ggml-metal/ggml-metal.metal`

**Quantized matmul has only one path: dequantize -> FP32×FP32.** No INT8 computation.

```metal
dequantize_q4_K(x, il, temp_a);           // Dequantize -> float4x4
simdgroup_multiply_accumulate(acc, ...);   // FP32 matmul
```

Non-quantized matmul: FP32×FP32 -> FP32. All Metal paths are FP32.

## 4. Dequantization Target Precision (Standalone Dequantization Functions)

In the matmul paths described above, CPU/CUDA do not go through full dequantization (they compute directly in integer domain). However, ggml also provides standalone dequantization functions, whose target precision varies by backend:

| Backend | Dequantization Target | Mechanism |
|---------|---------------------|-----------|
| CPU | Always **FP32** | Fixed function signature `float * y` |
| CUDA | **FP32 or FP16** | Template `dequantize_block_q4_K<dst_t>()` |
| Metal | **FP32 or FP16** | Template parameter `type4x4` (`float4x4` or `half4x4`) |

## 5. Precision Control

ggml provides runtime precision control via `dst->op_params[0]`:

| Value | Constant | Effect |
|-------|----------|--------|
| 0 | `GGML_PREC_DEFAULT` | Allow backend to use lower precision |
| 10 | `GGML_PREC_F32` | Force FP32 accumulation |

CPU and Metal always use FP32, so this switch primarily affects the CUDA cuBLAS path.

## 6. Precision Pitfall: Stronger Hardware Does Not Always Mean Higher Precision

### 6.1 CUDA cuBLAS: Consumer-grade Powerful GPUs Have Lower Accumulation Precision

| GPU | cuBLAS Accumulation Precision |
|-----|------------------------------|
| Older GPUs (no FP16 Tensor Cores) | FP32 |
| RTX 30xx/40xx (powerful consumer GPUs) | **FP16** <- lower |
| CDNA/RDNA4 (professional GPUs) | FP32 |

### 6.2 CUDA MMQ vs cuBLAS: Batch Size Affects Precision Path

The same quantized model on the same GPU may take different precision paths depending on batch size:
- Small batch -> MMQ (INT8, activations are quantized)
- Large batch -> cuBLAS (FP16 Tensor Cores, activations not quantized but accumulation may be FP16)

## 7. Three-Backend Comparison Overview

| | CPU | CUDA (MMQ) | CUDA (cuBLAS fallback) | Metal |
|--|-----|-----------|----------------------|-------|
| Quantized matmul method | Integer vec_dot | INT8xINT8 dp4a | Dequantize -> FP16/FP32 | Dequantize -> FP32 |
| Multiply precision | Integer multiply (INT8 value range) | INT8xINT8 | FP16 or FP32 | FP32 |
| Accumulate precision | INT32 -> FP32 | INT32 -> FP32 | FP16 or FP32 | FP32 |
| Activation handling | FP32 -> Q8 (INT8) | FP32 -> Q8 (INT8) | Stays FP32 | Stays FP32 |
| Alternative path | None | <-> cuBLAS (auto) | <-> MMQ (auto) | None |
| User control | No | `GGML_CUDA_FORCE_MMQ` | `GGML_CUDA_FORCE_CUBLAS` | No |
| Precision characteristics | Activation quantization loss | Activation quantization loss | Large batch may use FP16 accumulation | Highest precision |
| Output precision | FP32 | FP32 | FP32 | FP32 |
