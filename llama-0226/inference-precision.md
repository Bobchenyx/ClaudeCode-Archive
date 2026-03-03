# llama.cpp 推理精度分析（CPU / CUDA / Metal）

本文分析 llama.cpp 中量化模型推理时 `ggml_mul_mat()`（ggml 的矩阵乘法算子，Transformer 中所有线性层的核心计算）的精度行为。

## 1. 量化 matmul 的完整 pipeline

### 1.1 CPU / CUDA：动态 WxA8

CPU 和 CUDA 在量化 matmul 中会将 FP32 激活值**临时在线量化为 Q8（INT8）**，使两侧都变成整数后用 INT8×INT8 硬件指令计算：

```
权重（Q4_K，4-bit 存储）                  激活值（FP32）
        │                                      │
   解包 4-bit → INT8                  quantize_row_q8_K()
   （纯位操作，scale 未参与）           FP32 → INT8 在线量化
        │                                      │
        ▼                                      ▼
   INT8（权重侧）                         INT8（激活侧）
        │                                      │
        └────────── INT8 × INT8 ───────────────┘
                        │
                   INT32 累加
                        │
                × FP32 scale（Q4_K.d × Q8_K.d）
                        │
                   FP32 输出
```

权重侧的 4-bit → INT8 是**解包**（纯位操作，值不变），不是反量化。激活侧的 FP32 → INT8 是真正的量化（有精度损失）。

### 1.2 Metal：反量化到 FP32

Metal 不走 INT8 路径。它将量化权重**反量化到 FP32**（应用 scale 恢复浮点值），激活值保持 FP32 原样：

```
权重（Q4_K，4-bit 存储）                  激活值（FP32）
        │                                      │
  dequantize_q4_K()                        保持原样
  反量化到 FP32（应用 scale）                   │
        │                                      │
        ▼                                      ▼
   FP32（权重侧）                         FP32（激活侧）
        │                                      │
        └────────── FP32 × FP32 ───────────────┘
                        │
                   FP32 累加
                        │
                   FP32 输出
```

原因：Apple GPU 没有高效的 INT8 matmul 指令，FP32 compute unit 更快。量化在 Metal 上只节省内存带宽，不省计算。

### 1.3 精度影响

| | CPU / CUDA（INT8 路径） | Metal（FP32 路径） |
|--|----------------------|------------------|
| 权重精度损失 | Q4_K 量化（相同） | Q4_K 量化（相同） |
| 激活精度损失 | FP32 → Q8 **有额外量化损失** | **无**（保持 FP32） |
| 总量化损失来源 | **两次**（权重 + 激活） | **一次**（仅权重） |

CPU/CUDA 用激活侧多一次量化损失，换取 INT8 整数指令的速度优势。Metal 精度更高但计算效率较低。这是**速度 vs 精度的 tradeoff**。

## 2. 激活值在线量化的具体实现

### 2.1 CPU 端

源码：`ggml/src/ggml-cpu/ggml-cpu.c` — `ggml_compute_forward_mul_mat()`

```c
// 行 1241：根据权重类型查表，得到配对的 Q8 类型
// Q4_K → vec_dot_type = Q8_K（硬编码在 type_traits_cpu 表中）
enum ggml_type const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;

// 行 1291-1326：FP32 激活值 → Q8 量化（无条件执行）
if (src1->type != vec_dot_type) {           // FP32 != Q8_K → 必定进入
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    from_float(src1->data, wdata, ne10);    // quantize_row_q8_K()
}
```

**没有任何分支选择"INT8 路径还是 FP32 路径"——代码结构本身只有这一条路。**

### 2.2 CUDA 端

源码：`ggml/src/ggml-cuda/mmq.cu`, `quantize.cu`

```cuda
// mmq.cu 行 73：断言激活值是 FP32
GGML_ASSERT(src1->type == GGML_TYPE_F32);

// mmq.cu 行 122-137：在 GPU 上分配缓冲区，将 FP32 激活量化为 Q8_1
ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), ...);
```

GPU 量化 kernel（`quantize.cu` 行 176-271）：

```cuda
const float4 xi = x4[...];                     // 加载 4 个 FP32
float amax = max(abs(xi.x), abs(xi.y), ...);   // 求最大绝对值
amax = warp_reduce_max(amax);                   // warp 内归约
const float d_inv = 127.0f / amax;              // 计算 scale
char4 q;
q.x = roundf(xi.x * d_inv);                    // FP32 → INT8
q.y = roundf(xi.y * d_inv);
q.z = roundf(xi.z * d_inv);
q.w = roundf(xi.w * d_inv);
yqs4[iqs/4] = q;                               // 写回 4 个 int8
```

### 2.3 Q8 激活量化函数（CPU 参考实现）

源码：`ggml/src/ggml-quants.c` — `quantize_row_q8_K_ref()`

```c
void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k) {
    for (int i = 0; i < nb; i++) {
        // 1. 找到 block（256 个值）的最大绝对值
        float amax = 0;
        for (int j = 0; j < QK_K; ++j)
            amax = MAX(amax, fabsf(x[j]));

        // 2. 计算 scale：映射 FP32 → [-127, 127]
        const float iscale = -127.f / max;

        // 3. 逐个量化 FP32 → INT8
        for (int j = 0; j < QK_K; ++j)
            y[i].qs[j] = MIN(127, nearest_int(iscale * x[j]));

        // 4. 存储 scale
        y[i].d = 1 / iscale;
    }
}
```

### 2.4 权重类型与 Q8 变体的配对

定义在 `type_traits_cpu[]`（`ggml-cpu.c` 行 207-390）：

| 权重类型 | 配对的 Q8 类型 | vec_dot 函数 |
|---------|-------------|-------------|
| Q4_0 | Q8_0（block 32） | `ggml_vec_dot_q4_0_q8_0` |
| Q4_1 | Q8_1（block 32） | `ggml_vec_dot_q4_1_q8_1` |
| Q5_0 | Q8_0（block 32） | `ggml_vec_dot_q5_0_q8_0` |
| Q5_1 | Q8_1（block 32） | `ggml_vec_dot_q5_1_q8_1` |
| Q2_K | Q8_K（block 256） | `ggml_vec_dot_q2_K_q8_K` |
| Q3_K | Q8_K（block 256） | `ggml_vec_dot_q3_K_q8_K` |
| Q4_K | Q8_K（block 256） | `ggml_vec_dot_q4_K_q8_K` |
| Q5_K | Q8_K（block 256） | `ggml_vec_dot_q5_K_q8_K` |
| Q6_K | Q8_K（block 256） | `ggml_vec_dot_q6_K_q8_K` |

Q8 有三种变体：

| 变体 | Block 大小 | Scale 精度 | 配对对象 |
|------|-----------|-----------|---------|
| Q8_0 | 32 | FP16 | Q4_0 / Q5_0 |
| Q8_1 | 32 | FP32 | Q4_1 / Q5_1 |
| Q8_K | 256 | FP32 | 所有 K-quant |

**Q8 的核心角色不是权重量化，而是激活值的临时在线量化格式。**

### 2.5 与传统 W8A8 的对比

llama.cpp 的做法本质是**动态 WxA8**，但与传统 W8A8 有关键区别：

| | 传统 W8A8 | llama.cpp |
|--|----------|-----------|
| 激活量化时机 | 离线/静态（校准数据集确定 scale） | **在线/动态**（每次 matmul 前实时算） |
| Scale 粒度 | per-tensor 或 per-channel | **per-block**（每 32 或 256 个值） |
| 层间传递 | INT8 在层间传递，误差**跨层累积** | 每层结束回到 FP32，误差**不跨层累积** |

```
传统 W8A8：
  → [INT8] → Layer1 → [INT8] → Layer2 → [INT8] →    误差跨层累积

llama.cpp：
  → [FP32] → Layer1 → [FP32] → Layer2 → [FP32] →    误差不跨层累积
               ↓ 临时Q8     ↓ 临时Q8
              算完回FP32   算完回FP32
```

llama.cpp 精度更高（动态 scale + 细粒度 + 不跨层累积），代价是每层多做一次量化操作。

### 2.6 Q8 量化的作用范围：常规 activation，不仅是 KV Cache

一个常见的误解是"激活量化只影响 KV Cache"。实际上，Q8 量化影响的是**每一个经过量化权重的线性层的 hidden state**。

Transformer 每层中有两类 `ggml_mul_mat`，只有第一类会触发 Q8 激活量化：

**类型 1：权重投影（触发 Q8 量化）**

```
src0 = 量化权重（Q4_K）    src1 = FP32 hidden state（常规 activation）

build_lora_mm(wq, cur)       → ggml_mul_mat(wq, cur)       // Q 投影
build_lora_mm(wk, cur)       → ggml_mul_mat(wk, cur)       // K 投影
build_lora_mm(wv, cur)       → ggml_mul_mat(wv, cur)       // V 投影
build_lora_mm(wo, attn_out)  → ggml_mul_mat(wo, attn_out)  // 输出投影
build_lora_mm(ffn_up, cur)   → ggml_mul_mat(ffn_up, cur)   // FFN up
build_lora_mm(ffn_gate, cur) → ggml_mul_mat(ffn_gate, cur) // FFN gate
build_lora_mm(ffn_down, cur) → ggml_mul_mat(ffn_down, cur) // FFN down
```

源码：`src/models/llama.cpp` 行 46-116

`src0`（权重）是量化类型 → `type_traits_cpu[src0->type].vec_dot_type` 查表得到 Q8_K → `src1`（hidden state）被量化为 Q8。**每层 7 个线性层，每个都会触发一次。**

**类型 2：Attention K×Q / V×KQ（默认不触发）**

```
src0 = K 或 V（来自 KV Cache，默认 F16）    src1 = Q 或 KQ（FP32/F16）

ggml_mul_mat(k, q)     // K×Q attention score    （llama-graph.cpp 行 1798）
ggml_mul_mat(v, kq)    // V×KQ attention output  （llama-graph.cpp 行 1842）
```

KV Cache 默认类型是 **F16**（`llama-context.cpp` 行 2792-2793），不是量化格式。`src0` 不是量化类型 → 不查 `type_traits_cpu` 量化表 → **不触发 Q8 激活量化**，直接走 FP 路径。

> 用户可通过 `--cache-type-k q8_0` / `--cache-type-v q8_0` 将 KV Cache 设为量化格式，此时 Attention 的 matmul 也会触发激活量化。但这是用户主动选择，非默认行为。

**总结：**

| matmul 类型 | src0 | src1 | 触发 Q8 量化？ |
|---|---|---|---|
| 权重投影（QKV/FFN 等，每层 7 次） | Q4_K（量化权重） | FP32 hidden state | **是，每次都触发** |
| Attention K×Q | F16（KV Cache，默认） | F16/FP32（Q 向量） | **否** |
| Attention V×KQ | F16（KV Cache，默认） | FP32（attention weights） | **否** |

## 3. 三个后端的 matmul 计算精度

### 3.1 CPU

源码：`ggml/src/ggml-cpu/quants.c`, `arch/x86/quants.c`, `arch/arm/quants.c`

**量化 matmul 唯一路径：整数乘法。** 没有 FP32 备选路径——不存在"如果 CPU 没有 INT8 指令就回退到 FP32 matmul"的分支。

**标量回退路径（无 SIMD 的 CPU）：** 任何 CPU 都能执行，因为只用了 C 语言基本整数运算：

```c
// Q4_0 × Q8_0 标量路径（quants.c 行 137-141）
const int v0 = (x[ib].qs[j] & 0x0F) - 8;   // 解包到 int（值域 [-8, 7]）
sumi0 += (v0 * y[ib].qs[j]);                // int × int8_t → int

// Q4_K × Q8_K 标量路径（quants.c 行 603-604）
aux16[l] = q8[l] * a[l];                    // int8_t × int8_t → int16_t
aux32[l] += scale * aux16[l];               // int32_t 累加
```

注意：标量路径中**值域**是 INT8 范围（-128~127），但 **C 类型**不全是 `int8_t`——Q4_0 路径将解包后的值存为 `int`（通常 32 位），乘法是 `int × int8_t`。这不影响结果正确性，因为值域不变。

**SIMD 加速路径：** VNNI、DOTPROD 等不是必需的，它们只影响**并行度和速度**，不改变计算逻辑：

| 指令集 | 乘法精度 | 累加精度 |
|-------|---------|---------|
| 无 SIMD（标量回退） | int × int8_t → int | int 累加 |
| SSSE3 `_mm_maddubs_epi16` | uint8 × int8 → INT16 | INT16 → INT32 |
| AVX512-VNNI `_mm256_dpbusd_epi32` | uint8 × int8 → INT32 | INT32 |
| AMX-INT8 | INT8 × INT8 → INT32 | INT32（矩阵级） |
| NEON+DOTPROD `vdotq_s32` | int8 × int8 → INT32 | INT32 |
| NEON（无 DOTPROD）`vmull_s8` | int8 × int8 → INT16 | INT16 → INT32 |
| SVE `svdot_s32` | int8 × int8 → INT32 | INT32 |

SIMD 路径在编译时通过 `__AVX512VNNI__`、`__ARM_FEATURE_DOTPROD__` 等宏选择，非运行时分发。

非量化 matmul：FP32×FP32 → FP32。

### 3.2 CUDA

源码：`ggml/src/ggml-cuda/ggml-cuda.cu`, `mmq.cu`, `vecdotq.cuh`, `quantize.cu`

#### 量化 matmul：MMQ 与 cuBLAS 自动选择

由 `ggml_cuda_should_use_mmq()`（`mmq.cu` 行 262-366）决定：

```
GPU 不支持 dp4a（compute capability < 6.1）
  → cuBLAS（反量化→FP16/FP32）

GPU 支持 dp4a 但没有 FP16 Tensor Core
  → MMQ（INT8×INT8）

GPU 同时有 dp4a 和 FP16 Tensor Core（RTX 20xx/30xx/40xx）
  → batch size < MMQ_DP4A_MAX_BATCH_SIZE → MMQ（INT8）
  → batch size 大 → cuBLAS（反量化→FP16，Tensor Core 吞吐更高）
```

编译时可强制覆盖：`GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS`。

#### MMQ 路径

```cuda
int sumi = 0;
sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);   // 4 组 int8×int8 → INT32
return d4 * (sumi * ds8f.x - ...);              // FP32 缩放
```

#### cuBLAS 路径（非量化 / 回退时的精度）

| 输入类型 | GPU 条件 | 乘法精度 | 累加精度 |
|---------|---------|---------|---------|
| FP16 | CDNA / RDNA4 | FP16 | **FP32** |
| FP16 | 其他消费级 GPU + `GGML_PREC_DEFAULT` | **FP16** | **FP16** |
| FP16 | 任意 + `GGML_PREC_F32` | FP16 | **FP32** |
| BF16 | 任意 | BF16 | **FP32** |
| FP32 | 任意 | FP32 | FP32 |

### 3.3 Metal

源码：`ggml/src/ggml-metal/ggml-metal.metal`

**量化 matmul 唯一路径：反量化→FP32×FP32。** 不做 INT8 计算。

```metal
dequantize_q4_K(x, il, temp_a);           // 反量化→float4x4
simdgroup_multiply_accumulate(acc, ...);   // FP32 matmul
```

非量化 matmul：FP32×FP32 → FP32。Metal 所有路径均为 FP32。

## 4. 反量化目标精度（独立反量化函数）

上述 matmul 中 CPU/CUDA 不经过完整反量化（整数域直接计算）。但 ggml 也提供独立反量化函数，目标精度因后端而异：

| 后端 | 反量化目标精度 | 机制 |
|------|-------------|------|
| CPU | 始终 **FP32** | 函数签名固定 `float * y` |
| CUDA | **FP32 或 FP16** | 模板 `dequantize_block_q4_K<dst_t>()` |
| Metal | **FP32 或 FP16** | 模板参数 `type4x4`（`float4x4` 或 `half4x4`） |

## 5. 精度控制

ggml 通过 `dst->op_params[0]` 提供运行时精度控制：

| 值 | 常量 | 效果 |
|----|------|------|
| 0 | `GGML_PREC_DEFAULT` | 允许后端使用低精度 |
| 10 | `GGML_PREC_F32` | 强制 FP32 累加 |

CPU 和 Metal 始终 FP32，此开关主要影响 CUDA cuBLAS 路径。

## 6. 精度陷阱：更强的硬件不一定精度更高

### 6.1 CUDA cuBLAS：消费级强卡累加精度更低

| GPU | cuBLAS 累加精度 |
|-----|---------------|
| 老 GPU（无 FP16 Tensor Core） | FP32 |
| RTX 30xx/40xx（消费级强卡） | **FP16** ← 更低 |
| CDNA/RDNA4（专业卡） | FP32 |

### 6.2 CUDA MMQ vs cuBLAS：batch size 影响精度路径

同一个量化模型在同一块 GPU 上，batch size 不同可能走不同精度路径：
- 小 batch → MMQ（INT8，激活被量化）
- 大 batch → cuBLAS（FP16 Tensor Core，激活不量化但累加可能 FP16）

## 7. 三后端对比总览

| | CPU | CUDA（MMQ） | CUDA（cuBLAS 回退） | Metal |
|--|-----|-----------|------------------|-------|
| 量化 matmul 方式 | 整数 vec_dot | INT8×INT8 dp4a | 反量化→FP16/FP32 | 反量化→FP32 |
| 乘法精度 | INT8 值域的整数乘法 | INT8×INT8 | FP16 或 FP32 | FP32 |
| 累加精度 | INT32 → FP32 | INT32 → FP32 | FP16 或 FP32 | FP32 |
| 激活值处理 | FP32 → Q8（INT8） | FP32 → Q8（INT8） | 保持 FP32 | 保持 FP32 |
| 备选路径 | 无 | ↔ cuBLAS（自动） | ↔ MMQ（自动） | 无 |
| 用户可控 | 否 | `GGML_CUDA_FORCE_MMQ` | `GGML_CUDA_FORCE_CUBLAS` | 否 |
| 精度特点 | 激活有量化损失 | 激活有量化损失 | 大 batch 可能 FP16 累加 | 最高精度 |
| 输出精度 | FP32 | FP32 | FP32 | FP32 |
