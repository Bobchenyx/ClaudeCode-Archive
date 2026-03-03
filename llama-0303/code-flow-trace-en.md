# llama.cpp Inference Code Flow Trace

Complete call path from `llama-cli` user input down to the ggml backend kernels.

Using **Llama architecture + Q4_K quantized weights** as example, tracing a single `ggml_mul_mat` (linear layer matrix multiplication) execution.

---

## Full Call Chain Overview

```
llama-cli main()
  +- cli_context::generate_completion()          <- Submit inference task
      +- server_context::start_loop()            <- Server loop processing
          +- llama_decode()                      <- Public C API
              +- llama_context::decode()         <- Split into micro-batches
                  +- llama_context::process_ubatch()
                      |-- model.build_graph()     <- Build computation graph (no compute)
                      |   +- llm_build_llama()    <- Llama architecture graph definition
                      |       |-- build_lora_mm(wq, cur)  <- Q projection
                      |       |   +- ggml_mul_mat()       <- Create MUL_MAT node
                      |       |-- build_lora_mm(wk, cur)  <- K projection
                      |       |-- build_lora_mm(wv, cur)  <- V projection
                      |       |-- build_attn(...)         <- KQV attention
                      |       +- build_ffn(...)           <- FFN (up/gate/down)
                      |           +- build_lora_mm(ffn_down, cur)
                      |               +- ggml_mul_mat()
                      |
                      +- graph_compute()         <- Execute computation graph
                          +- ggml_backend_sched_graph_compute_async()
                              +- ggml_backend_sched_compute_splits()
                                  +- ggml_backend_graph_compute_async()
                                      +- backend->iface.graph_compute()
                                          |-- [CPU]   ggml_backend_cpu_graph_compute()
                                          |-- [CUDA]  ggml_cuda_graph_compute()
                                          +- [Metal] ggml_metal_graph_compute()
```

---

## Step 1: CLI Entry Point

**File:** `tools/cli/cli.cpp`

```
main()                                          // line 198
  |-- common_params_parse(argc, argv, params)   // Parse command-line arguments
  |-- cli_context ctx_cli(params)               // line 216: Create CLI context
  |-- ctx_cli.ctx_server.load_model(params)     // line 243: Load model
  +- ctx_cli.ctx_server.start_loop()            // line 253: Enter main loop
```

User input triggers `generate_completion()` (line 409), which creates a `SERVER_TASK_TYPE_COMPLETION` task and sends it to the embedded server:

```cpp
// lines 74-94
std::string generate_completion(result_timings & out_timings) {
    server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
    // ... set task parameters ...
    rd.post_task({std::move(task)});  // Submit to server queue
}
```

---

## Step 2: Server Dispatch -> llama_decode

**File:** `tools/server/server-context.cpp`

The server loop processes batches and calls the core inference API:

```cpp
const int ret = llama_decode(ctx, batch_view);  // line 2622
```

---

## Step 3: llama_decode -- Public C API Entry

**File:** `src/llama-context.cpp`

```cpp
int32_t llama_decode(                           // line 3302
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);         // line 3305: Forward to C++ implementation
    return ret;
}
```

---

## Step 4: llama_context::decode -- Micro-batch Loop

**File:** `src/llama-context.cpp` (line 1421)

Splits large batch into micro-batches (ubatch) and processes each:

```
llama_context::decode(batch)                    // line 1421
  |-- Initialize batch allocator
  +- for each ubatch:
       +- process_ubatch(ubatch, ...)           // line 1580
```

---

## Step 5: process_ubatch -- Graph Build + Graph Compute

**File:** `src/llama-context.cpp` (line 1070)

This is the **core of the two-phase execution**: first build the computation graph, then execute it.

```cpp
llm_graph_result * llama_context::process_ubatch(...) {
    // Phase 1: Build computation graph (deferred execution, only records operations)
    gf = model.build_graph(gparams);            // line 1096

    // Allocate backend memory for all tensors in the graph
    ggml_backend_sched_alloc_graph(sched, gf);  // line 1106

    // Fill input data (token embeddings, etc.)
    res->set_inputs(&ubatch);                   // line 1117

    // Phase 2: Execute computation graph
    graph_compute(res->get_gf(), ...);          // line 1122
}
```

---

## Step 6: build_graph -- Architecture Dispatch

**File:** `src/llama-model.cpp` (line 8357)

Selects the graph builder based on model architecture:

```cpp
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    switch (arch) {
        case LLM_ARCH_LLAMA:
            llm = std::make_unique<llm_build_llama<false>>(*this, params);  // line 8363
            break;
        case LLM_ARCH_LLAMA4:      // ...
        case LLM_ARCH_FALCON:      // ...
        case LLM_ARCH_QWEN2:       // ...
        // ... 112+ architectures ...
    }
    return llm->build(...);
}
```

---

## Step 7: llm_build_llama -- Transformer Layer Definition

**File:** `src/models/llama.cpp` (line 4)

Defines the entire Transformer computation graph layer by layer in the constructor:

```cpp
llm_build_llama(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params)
{
    inpL = build_inp_embd(model.tok_embd);      // Token embedding

    for (int il = 0; il < n_layer; ++il) {
        // === Self-Attention ===
        cur = build_norm(inpL, model.layers[il].attn_norm, ...);  // RMSNorm

        Qcur = build_lora_mm(model.layers[il].wq, cur);  // line 46: Q projection <- ggml_mul_mat
        Kcur = build_lora_mm(model.layers[il].wk, cur);  // line 52: K projection <- ggml_mul_mat
        Vcur = build_lora_mm(model.layers[il].wv, cur);  // line 58: V projection <- ggml_mul_mat

        // RoPE
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, ...);  // line 68
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, ...);  // line 74

        // KQV attention
        cur = build_attn(inp_attn, model.layers[il].wo, ...,       // line 91
                          Qcur, Kcur, Vcur, ...);  // also contains ggml_mul_mat internally

        // === FFN ===
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, ...);  // line 106

        cur = build_ffn(cur,                               // line 111
                model.layers[il].ffn_up,    ...,  // up projection   <- ggml_mul_mat
                model.layers[il].ffn_gate,  ...,  // gate projection <- ggml_mul_mat
                model.layers[il].ffn_down,  ...,  // down projection <- ggml_mul_mat
                LLM_FFN_SILU, LLM_FFN_PAR, il);

        inpL = ggml_add(ctx0, cur, ffn_inp);     // Residual connection
    }

    // Final norm + output projection
    cur = build_norm(cur, model.output_norm, ...);
    cur = build_lora_mm(model.output, cur);       // output projection <- ggml_mul_mat
}
```

---

## Step 8: build_lora_mm -> ggml_mul_mat -- Create Graph Node

**File:** `src/llama-graph.cpp` (line 900)

```cpp
ggml_tensor * llm_graph_context::build_lora_mm(ggml_tensor * w, ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);     // line 903
    // ... LoRA adapter handling (if any) ...
    return res;
}
```

**File:** `ggml/src/ggml.c` (line 3201)

```cpp
struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx,
                                  struct ggml_tensor * a,    // weights (Q4_K)
                                  struct ggml_tensor * b) {  // activations (FP32)
    // No computation happens here! Only creates a graph node
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op     = GGML_OP_MUL_MAT;    // line 3211: Mark operation type
    result->src[0] = a;                   // line 3212: weights
    result->src[1] = b;                   // line 3213: activations
    return result;
}
```

> **Key concept:** Up to this point, we are only "drawing the blueprint" -- no actual computation has occurred. All `ggml_mul_mat` / `ggml_add` / `ggml_rope_ext` calls merely add nodes to the computation graph.

---

## Step 9: graph_compute -- Execute Computation Graph

**File:** `src/llama-context.cpp` (line 2055)

```cpp
ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    // Set thread count
    // ...
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);  // line 2074
    return status;
}
```

---

## Step 10: Backend Scheduler -- Graph Splitting and Dispatch

**File:** `ggml/src/ggml-backend.cpp`

The scheduler splits the computation graph by backend, sending each split to the corresponding backend for execution:

```
ggml_backend_sched_graph_compute_async()        // line 1793
  +- ggml_backend_sched_compute_splits()        // line 1805
       +- for each split:                       // line 1453
            |-- Copy input tensors to target backend
            +- ggml_backend_graph_compute_async(split_backend, &split->graph)  // line 1582
                 +- backend->iface.graph_compute(backend, cgraph)    // Function pointer dispatch
```

> If the model is distributed across multiple backends (e.g., some layers on GPU, some on CPU), the scheduler automatically transfers data between backends.

---

## Step 11a: CPU Backend Execution Path

```
backend->iface.graph_compute()
  +- ggml_backend_cpu_graph_compute()           // ggml-cpu.cpp line 170
      +- ggml_graph_compute(cgraph, &cplan)     // ggml-cpu.c line 3207
          +- Multi-threaded iteration over each graph node
              +- ggml_compute_forward(params, node)
                  +- case GGML_OP_MUL_MAT:     // ggml-cpu.c line 1805
                      +- ggml_compute_forward_mul_mat(params, tensor)  // line 1807
```

### ggml_compute_forward_mul_mat (line 1229)

**This is the core CPU matmul function.** For Q4_K weights:

```
ggml_compute_forward_mul_mat()                  // ggml-cpu.c line 1229
  |
  |-- Table lookup for compute functions                     // line 1241
  |   vec_dot_type = type_traits_cpu[Q4_K].vec_dot_type  -> Q8_K
  |   vec_dot      = type_traits_cpu[Q4_K].vec_dot       -> ggml_vec_dot_q4_K_q8_K
  |   from_float   = type_traits_cpu[Q8_K].from_float    -> quantize_row_q8_K
  |
  |-- Online activation quantization                         // line 1291
  |   if (src1->type != Q8_K)                   // FP32 != Q8_K -> always enters
  |       from_float(src1->data, wdata, ne10)   // FP32 -> Q8_K (INT8)
  |
  +- Execute quantized dot product                           // line 1400+
      for each row:
          vec_dot(...)                           // -> ggml_vec_dot_q4_K_q8_K
```

### vec_dot Final Kernel

**File:** `ggml/src/ggml-cpu/quants.c` (line 550)

```
ggml_vec_dot_q4_K_q8_K_generic()               // Scalar fallback
  |-- Unpack Q4_K: q4[l] & 0xF -> int8_t       // line 585: 4-bit -> INT8
  |-- Integer multiply: q8[l] * a[l] -> int16_t // line 603
  |-- Integer accumulate: aux32[l] += scale * aux16[l]  // line 604
  +- Float scaling: sums[l] += d * aux32[l]     // line 617

With SIMD acceleration (compile-time selection):
  |-- x86 AVX2:    _mm256_maddubs_epi16          // arch/x86/quants.c
  |-- x86 VNNI:    _mm256_dpbusd_epi32
  |-- ARM NEON:    vmull_s8 / vdotq_s32          // arch/arm/quants.c
  +- ...
```

---

## Step 11b: CUDA Backend Execution Path

```
backend->iface.graph_compute()
  +- ggml_cuda_graph_compute()
      +- Iterate over each graph node
          +- case GGML_OP_MUL_MAT:             // ggml-cuda.cu line 2619
              +- ggml_cuda_mul_mat()            // line 2183
```

### ggml_cuda_mul_mat -- Path Selection (line 2183)

```
ggml_cuda_mul_mat(ctx, src0, src1, dst)
  |
  |-- Condition checks:
  |   use_mul_mat_vec_f  = non-quantized + batch=1    // FP32/FP16/BF16 vector multiply
  |   use_mul_mat_f      = non-quantized              // FP matrix multiply
  |   use_mul_mat_vec_q  = quantized + batch <= 8     // Quantized vector multiply (MMVQ)
  |   use_mul_mat_q      = quantized + should_use_mmq()  // Quantized matrix multiply (MMQ)
  |
  |-- Path selection (lines 2243-2262):
  |   if (use_mul_mat_vec_f)    -> ggml_cuda_mul_mat_vec_f()       // line 2246
  |   elif (use_mul_mat_f)      -> ggml_cuda_mul_mat_f()           // line 2248
  |   elif (use_mul_mat_vec_q)  -> ggml_cuda_mul_mat_vec_q()       // line 2250
  |   elif (use_mul_mat_q)      -> ggml_cuda_mul_mat_q()           // line 2252: MMQ path
  |   elif (batched_cublas)     -> ggml_cuda_mul_mat_batched_cublas()  // line 2256
  |   else (split multi-GPU)    -> ggml_cuda_op_mul_mat(... quantize_mmq_q8_1_cuda)  // line 2262
```

### MMQ Path (Quantized Q4_K matmul)

**File:** `ggml/src/ggml-cuda/mmq.cu` (line 71)

```
ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst)  // line 71
  |
  |-- GGML_ASSERT(src1->type == GGML_TYPE_F32)   // line 73: Activations must be FP32
  |
  |-- Allocate Q8_1 buffer                        // lines 122-124
  |   ggml_cuda_pool_alloc<char> src1_q8_1(...)
  |
  |-- Quantize activations FP32 -> Q8_1 on GPU   // line 136
  |   quantize_mmq_q8_1_cuda(src1_d, ..., src1_q8_1.get(), ...)
  |   +- quantize_mmq_q8_1 kernel                // quantize.cu line 176
  |       |-- float4 xi = x4[...]                 // Load 4 FP32 values
  |       |-- amax = warp_reduce_max(...)         // Find max
  |       |-- d_inv = 127.0f / amax              // Scale
  |       +- q.x = roundf(xi.x * d_inv)         // FP32 -> INT8
  |
  +- Launch MMQ kernel                            // line 156
      ggml_cuda_mul_mat_q_switch_type(ctx, args, stream)  // mmq.cu line 6
        +- case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(...)
              +- dp4a(vi0, u[i], sumi)           // INT8 x INT8 -> INT32
                  +- __dp4a(a, b, c)             // common.cuh line 696: Hardware intrinsic
```

### cuBLAS Fallback Path

When `should_use_mmq()` returns false (e.g., large batch + Tensor Cores available):

```
ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, ...)
  +- ggml_cuda_op_mul_mat_cublas()               // ggml-cuda.cu line 1228
      |-- Dequantize weights Q4_K -> FP16         // dequantize kernel
      +- cublasGemmEx(... CUBLAS_COMPUTE_16F)    // FP16 Tensor Core matmul
```

---

## Step 11c: Metal Backend Execution Path

```
backend->iface.graph_compute()
  +- ggml_metal_graph_compute()                  // ggml-metal.cpp
      +- Iterate over each graph node
          +- case GGML_OP_MUL_MAT:              // ggml-metal-ops.cpp line 340
              +- ggml_metal_op_mul_mat()         // line 1931
```

### ggml_metal_op_mul_mat (line 1931)

```
ggml_metal_op_mul_mat()
  |-- Select kernel pipeline (based on type, dimensions, batch size)
  |   e.g.: kernel_mul_mv_ext_q4_K_f32
  |
  +- Execute Metal compute shader on GPU:
      |-- dequantize_q4_K(x, il, temp_a)         // Dequantize -> float4x4 (FP32)
      +- dot(lx[ch], y4x4[ir1][...])            // FP32 x FP32 dot product
```

**Metal kernel source:** `ggml/src/ggml-metal/ggml-metal.metal`

```metal
// line 3339: Quantized matmul kernel template
template<short r1ptg, typename q_t, short chpb,
         void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
void kernel_mul_mv_ext_q4x4_f32_impl(...) {
    float4x4 lx[chpt];                          // FP32 temporary
    deq_t4x4(xq, cch, lx[ch]);                  // line 3386: Dequantize to FP32
    sumf[ir1] +=
        dot(lx[ch][0], y4x4[ir1][...][0]) +     // line 3399: FP32 dot product
        dot(lx[ch][1], y4x4[ir1][...][1]) +
        dot(lx[ch][2], y4x4[ir1][...][2]) +
        dot(lx[ch][3], y4x4[ir1][...][3]);
}
```

---

## Appendix: Key File Index

| Layer | File | Responsibility |
|-------|------|----------------|
| CLI | `tools/cli/cli.cpp` | User interaction, task submission |
| Server | `tools/server/server-context.cpp` | Batch scheduling, calls `llama_decode` |
| API | `src/llama-context.cpp` | `llama_decode()` -> `decode()` -> `process_ubatch()` |
| Model | `src/llama-model.cpp` | `build_graph()` architecture dispatch |
| Graph def | `src/models/llama.cpp` | Llama Transformer layer graph structure |
| Graph utils | `src/llama-graph.cpp` | `build_lora_mm()` / `build_ffn()` / `build_attn()` |
| ggml node | `ggml/src/ggml.c` | `ggml_mul_mat()` creates graph nodes |
| Scheduler | `ggml/src/ggml-backend.cpp` | Graph splitting, backend dispatch |
| CPU compute | `ggml/src/ggml-cpu/ggml-cpu.c` | `ggml_compute_forward_mul_mat()` |
| CPU kernel | `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_q4_K_q8_K_generic()` |
| CPU SIMD | `ggml/src/ggml-cpu/arch/x86/quants.c` | x86 SIMD-accelerated vec_dot |
| CUDA dispatch | `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat()` path selection |
| CUDA MMQ | `ggml/src/ggml-cuda/mmq.cu` | Quantized matrix multiply kernel |
| CUDA quantize | `ggml/src/ggml-cuda/quantize.cu` | GPU activation quantization FP32->Q8_1 |
| CUDA dp4a | `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_dp4a()` INT8 x INT8 |
| Metal dispatch | `ggml/src/ggml-metal/ggml-metal-ops.cpp` | `ggml_metal_op_mul_mat()` |
| Metal kernel | `ggml/src/ggml-metal/ggml-metal.metal` | Dequantize + FP32 matmul shader |
