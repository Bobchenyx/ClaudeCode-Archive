# llama.cpp Inference Code Flow Trace

从 `llama-cli` 用户输入到 ggml 底层 kernel 的完整调用路径。

以 **Llama 架构 + Q4_K 量化权重** 为例，追踪一次 `ggml_mul_mat`（线性层矩阵乘法）的执行。

---

## 完整调用链总览

```
llama-cli main()
  └─ cli_context::generate_completion()          ← 提交推理任务
      └─ server_context::start_loop()            ← 服务端循环处理
          └─ llama_decode()                      ← 公共 C API
              └─ llama_context::decode()         ← 切分 micro-batch
                  └─ llama_context::process_ubatch()
                      ├─ model.build_graph()     ← 构建计算图（不计算）
                      │   └─ llm_build_llama()   ← Llama 架构的图定义
                      │       ├─ build_lora_mm(wq, cur)  ← Q 投影
                      │       │   └─ ggml_mul_mat()      ← 创建 MUL_MAT 节点
                      │       ├─ build_lora_mm(wk, cur)  ← K 投影
                      │       ├─ build_lora_mm(wv, cur)  ← V 投影
                      │       ├─ build_attn(...)         ← KQV 注意力
                      │       └─ build_ffn(...)          ← FFN（up/gate/down）
                      │           └─ build_lora_mm(ffn_down, cur)
                      │               └─ ggml_mul_mat()
                      │
                      └─ graph_compute()         ← 执行计算图
                          └─ ggml_backend_sched_graph_compute_async()
                              └─ ggml_backend_sched_compute_splits()
                                  └─ ggml_backend_graph_compute_async()
                                      └─ backend->iface.graph_compute()
                                          ├─ [CPU]   ggml_backend_cpu_graph_compute()
                                          ├─ [CUDA]  ggml_cuda_graph_compute()
                                          └─ [Metal] ggml_metal_graph_compute()
```

---

## Step 1: CLI 入口

**文件：** `tools/cli/cli.cpp`

```
main()                                          // 行 198
  ├─ common_params_parse(argc, argv, params)    // 解析命令行参数
  ├─ cli_context ctx_cli(params)                // 行 216: 创建 CLI 上下文
  ├─ ctx_cli.ctx_server.load_model(params)      // 行 243: 加载模型
  └─ ctx_cli.ctx_server.start_loop()            // 行 253: 进入主循环
```

用户输入触发 `generate_completion()`（行 409），它创建一个 `SERVER_TASK_TYPE_COMPLETION` 任务发送给内嵌的 server：

```cpp
// 行 74-94
std::string generate_completion(result_timings & out_timings) {
    server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
    // ... 设置任务参数 ...
    rd.post_task({std::move(task)});  // 提交到 server 队列
}
```

---

## Step 2: Server 调度 → llama_decode

**文件：** `tools/server/server-context.cpp`

Server 循环中处理 batch，调用核心推理 API：

```cpp
const int ret = llama_decode(ctx, batch_view);  // 行 2622
```

---

## Step 3: llama_decode — 公共 C API 入口

**文件：** `src/llama-context.cpp`

```cpp
int32_t llama_decode(                           // 行 3302
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);         // 行 3305: 转发到 C++ 实现
    return ret;
}
```

---

## Step 4: llama_context::decode — Micro-batch 循环

**文件：** `src/llama-context.cpp`（行 1421）

将大 batch 切分为 micro-batch（ubatch），逐个处理：

```
llama_context::decode(batch)                    // 行 1421
  ├─ 初始化 batch allocator
  └─ for each ubatch:
       └─ process_ubatch(ubatch, ...)           // 行 1580
```

---

## Step 5: process_ubatch — 图构建 + 图计算

**文件：** `src/llama-context.cpp`（行 1070）

这是**两阶段执行的核心**：先构建计算图，再执行。

```cpp
llm_graph_result * llama_context::process_ubatch(...) {
    // 阶段 1: 构建计算图（延迟执行，只记录操作）
    gf = model.build_graph(gparams);            // 行 1096

    // 为图中所有张量分配后端内存
    ggml_backend_sched_alloc_graph(sched, gf);  // 行 1106

    // 填充输入数据（token embedding 等）
    res->set_inputs(&ubatch);                   // 行 1117

    // 阶段 2: 执行计算图
    graph_compute(res->get_gf(), ...);          // 行 1122
}
```

---

## Step 6: build_graph — 架构分发

**文件：** `src/llama-model.cpp`（行 8357）

根据模型架构选择对应的图构建器：

```cpp
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    switch (arch) {
        case LLM_ARCH_LLAMA:
            llm = std::make_unique<llm_build_llama<false>>(*this, params);  // 行 8363
            break;
        case LLM_ARCH_LLAMA4:      // ...
        case LLM_ARCH_FALCON:      // ...
        case LLM_ARCH_QWEN2:       // ...
        // ... 112+ 架构 ...
    }
    return llm->build(...);
}
```

---

## Step 7: llm_build_llama — Transformer 层定义

**文件：** `src/models/llama.cpp`（行 4）

在构造函数中逐层定义整个 Transformer 的计算图：

```cpp
llm_build_llama(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params)
{
    inpL = build_inp_embd(model.tok_embd);      // Token embedding

    for (int il = 0; il < n_layer; ++il) {
        // === Self-Attention ===
        cur = build_norm(inpL, model.layers[il].attn_norm, ...);  // RMSNorm

        Qcur = build_lora_mm(model.layers[il].wq, cur);  // 行 46: Q 投影 ← ggml_mul_mat
        Kcur = build_lora_mm(model.layers[il].wk, cur);  // 行 52: K 投影 ← ggml_mul_mat
        Vcur = build_lora_mm(model.layers[il].wv, cur);  // 行 58: V 投影 ← ggml_mul_mat

        // RoPE
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, ...);  // 行 68
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, ...);  // 行 74

        // KQV attention
        cur = build_attn(inp_attn, model.layers[il].wo, ...,       // 行 91
                          Qcur, Kcur, Vcur, ...);  // 内部也有 ggml_mul_mat

        // === FFN ===
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, ...);  // 行 106

        cur = build_ffn(cur,                               // 行 111
                model.layers[il].ffn_up,    ...,  // up 投影   ← ggml_mul_mat
                model.layers[il].ffn_gate,  ...,  // gate 投影 ← ggml_mul_mat
                model.layers[il].ffn_down,  ...,  // down 投影 ← ggml_mul_mat
                LLM_FFN_SILU, LLM_FFN_PAR, il);

        inpL = ggml_add(ctx0, cur, ffn_inp);     // 残差连接
    }

    // Final norm + output projection
    cur = build_norm(cur, model.output_norm, ...);
    cur = build_lora_mm(model.output, cur);       // output 投影 ← ggml_mul_mat
}
```

---

## Step 8: build_lora_mm → ggml_mul_mat — 创建图节点

**文件：** `src/llama-graph.cpp`（行 900）

```cpp
ggml_tensor * llm_graph_context::build_lora_mm(ggml_tensor * w, ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);     // 行 903
    // ... LoRA adapter 处理（如有）...
    return res;
}
```

**文件：** `ggml/src/ggml.c`（行 3201）

```cpp
struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx,
                                  struct ggml_tensor * a,    // 权重（Q4_K）
                                  struct ggml_tensor * b) {  // 激活值（FP32）
    // 此时不计算！只创建一个图节点
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op     = GGML_OP_MUL_MAT;    // 行 3211: 标记操作类型
    result->src[0] = a;                   // 行 3212: 权重
    result->src[1] = b;                   // 行 3213: 激活值
    return result;
}
```

> **关键概念：** 到此为止只是"画蓝图"，没有任何实际计算发生。所有 `ggml_mul_mat` / `ggml_add` / `ggml_rope_ext` 等调用都只是在计算图中添加节点。

---

## Step 9: graph_compute — 执行计算图

**文件：** `src/llama-context.cpp`（行 2055）

```cpp
ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    // 设置线程数
    // ...
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);  // 行 2074
    return status;
}
```

---

## Step 10: Backend Scheduler — 图切分与分发

**文件：** `ggml/src/ggml-backend.cpp`

调度器将计算图按后端切分（split），每个 split 发送给对应后端执行：

```
ggml_backend_sched_graph_compute_async()        // 行 1793
  └─ ggml_backend_sched_compute_splits()        // 行 1805
       └─ for each split:                       // 行 1453
            ├─ 拷贝输入张量到目标后端
            └─ ggml_backend_graph_compute_async(split_backend, &split->graph)  // 行 1582
                 └─ backend->iface.graph_compute(backend, cgraph)    // 函数指针分发
```

> 如果模型分布在多个后端（如部分层在 GPU、部分在 CPU），调度器会自动在后端间搬运数据。

---

## Step 11a: CPU 后端执行路径

```
backend->iface.graph_compute()
  └─ ggml_backend_cpu_graph_compute()           // ggml-cpu.cpp 行 170
      └─ ggml_graph_compute(cgraph, &cplan)     // ggml-cpu.c 行 3207
          └─ 多线程遍历图中每个节点
              └─ ggml_compute_forward(params, node)
                  └─ case GGML_OP_MUL_MAT:     // ggml-cpu.c 行 1805
                      └─ ggml_compute_forward_mul_mat(params, tensor)  // 行 1807
```

### ggml_compute_forward_mul_mat（行 1229）

**这是 CPU matmul 的核心函数。** 对于 Q4_K 权重：

```
ggml_compute_forward_mul_mat()                  // ggml-cpu.c 行 1229
  │
  ├─ 查表获取计算函数                            // 行 1241
  │   vec_dot_type = type_traits_cpu[Q4_K].vec_dot_type  → Q8_K
  │   vec_dot      = type_traits_cpu[Q4_K].vec_dot       → ggml_vec_dot_q4_K_q8_K
  │   from_float   = type_traits_cpu[Q8_K].from_float    → quantize_row_q8_K
  │
  ├─ 激活值在线量化                              // 行 1291
  │   if (src1->type != Q8_K)                   // FP32 ≠ Q8_K → 必定进入
  │       from_float(src1->data, wdata, ne10)   // FP32 → Q8_K（INT8）
  │
  └─ 执行量化点积                               // 行 1400+
      for each row:
          vec_dot(...)                           // → ggml_vec_dot_q4_K_q8_K
```

### vec_dot 最终 kernel

**文件：** `ggml/src/ggml-cpu/quants.c`（行 550）

```
ggml_vec_dot_q4_K_q8_K_generic()               // 标量回退
  ├─ 解包 Q4_K：q4[l] & 0xF → int8_t          // 行 585: 4-bit → INT8
  ├─ 整数乘法：q8[l] * a[l] → int16_t         // 行 603
  ├─ 整数累加：aux32[l] += scale * aux16[l]    // 行 604
  └─ 浮点缩放：sums[l] += d * aux32[l]        // 行 617

如有 SIMD 加速（编译时选择）：
  ├─ x86 AVX2:    _mm256_maddubs_epi16          // arch/x86/quants.c
  ├─ x86 VNNI:    _mm256_dpbusd_epi32
  ├─ ARM NEON:    vmull_s8 / vdotq_s32          // arch/arm/quants.c
  └─ ...
```

---

## Step 11b: CUDA 后端执行路径

```
backend->iface.graph_compute()
  └─ ggml_cuda_graph_compute()
      └─ 遍历图中每个节点
          └─ case GGML_OP_MUL_MAT:             // ggml-cuda.cu 行 2619
              └─ ggml_cuda_mul_mat()            // 行 2183
```

### ggml_cuda_mul_mat — 路径选择（行 2183）

```
ggml_cuda_mul_mat(ctx, src0, src1, dst)
  │
  ├─ 判断条件：
  │   use_mul_mat_vec_f  = 非量化 + batch=1    // FP32/FP16/BF16 向量乘
  │   use_mul_mat_f      = 非量化              // FP 矩阵乘
  │   use_mul_mat_vec_q  = 量化 + batch ≤ 8    // 量化向量乘（MMVQ）
  │   use_mul_mat_q      = 量化 + should_use_mmq()  // 量化矩阵乘（MMQ）
  │
  ├─ 路径选择（行 2243-2262）：
  │   if (use_mul_mat_vec_f)    → ggml_cuda_mul_mat_vec_f()       // 行 2246
  │   elif (use_mul_mat_f)      → ggml_cuda_mul_mat_f()           // 行 2248
  │   elif (use_mul_mat_vec_q)  → ggml_cuda_mul_mat_vec_q()       // 行 2250
  │   elif (use_mul_mat_q)      → ggml_cuda_mul_mat_q()           // 行 2252: MMQ 路径
  │   elif (batched_cublas)     → ggml_cuda_mul_mat_batched_cublas()  // 行 2256
  │   else (split 多卡回退)     → ggml_cuda_op_mul_mat(... quantize_mmq_q8_1_cuda)  // 行 2262
```

### MMQ 路径（量化 Q4_K matmul）

**文件：** `ggml/src/ggml-cuda/mmq.cu`（行 71）

```
ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst)  // 行 71
  │
  ├─ GGML_ASSERT(src1->type == GGML_TYPE_F32)   // 行 73: 激活必须是 FP32
  │
  ├─ 分配 Q8_1 缓冲区                            // 行 122-124
  │   ggml_cuda_pool_alloc<char> src1_q8_1(...)
  │
  ├─ GPU 上量化激活值 FP32 → Q8_1               // 行 136
  │   quantize_mmq_q8_1_cuda(src1_d, ..., src1_q8_1.get(), ...)
  │   └─ quantize_mmq_q8_1 kernel               // quantize.cu 行 176
  │       ├─ float4 xi = x4[...]                 // 加载 4 个 FP32
  │       ├─ amax = warp_reduce_max(...)         // 求 max
  │       ├─ d_inv = 127.0f / amax              // scale
  │       └─ q.x = roundf(xi.x * d_inv)        // FP32 → INT8
  │
  └─ 启动 MMQ kernel                            // 行 156
      ggml_cuda_mul_mat_q_switch_type(ctx, args, stream)  // mmq.cu 行 6
        └─ case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(...)
              └─ dp4a(vi0, u[i], sumi)           // INT8×INT8 → INT32
                  └─ __dp4a(a, b, c)             // common.cuh 行 696: 硬件指令
```

### cuBLAS 回退路径

当 `should_use_mmq()` 返回 false（例如大 batch + Tensor Core 可用）：

```
ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, ...)
  └─ ggml_cuda_op_mul_mat_cublas()               // ggml-cuda.cu 行 1228
      ├─ 反量化权重 Q4_K → FP16                   // dequantize kernel
      └─ cublasGemmEx(... CUBLAS_COMPUTE_16F)    // FP16 Tensor Core matmul
```

---

## Step 11c: Metal 后端执行路径

```
backend->iface.graph_compute()
  └─ ggml_metal_graph_compute()                  // ggml-metal.cpp
      └─ 遍历图中每个节点
          └─ case GGML_OP_MUL_MAT:              // ggml-metal-ops.cpp 行 340
              └─ ggml_metal_op_mul_mat()         // 行 1931
```

### ggml_metal_op_mul_mat（行 1931）

```
ggml_metal_op_mul_mat()
  ├─ 选择 kernel pipeline（基于类型、维度、batch size）
  │   例如: kernel_mul_mv_ext_q4_K_f32
  │
  └─ 在 GPU 上执行 Metal compute shader:
      ├─ dequantize_q4_K(x, il, temp_a)         // 反量化 → float4x4（FP32）
      └─ dot(lx[ch], y4x4[ir1][...])            // FP32 × FP32 点积
```

**Metal kernel 源码：** `ggml/src/ggml-metal/ggml-metal.metal`

```metal
// 行 3339: 量化矩阵乘 kernel 模板
template<short r1ptg, typename q_t, short chpb,
         void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
void kernel_mul_mv_ext_q4x4_f32_impl(...) {
    float4x4 lx[chpt];                          // FP32 暂存
    deq_t4x4(xq, cch, lx[ch]);                  // 行 3386: 反量化到 FP32
    sumf[ir1] +=
        dot(lx[ch][0], y4x4[ir1][...][0]) +     // 行 3399: FP32 dot product
        dot(lx[ch][1], y4x4[ir1][...][1]) +
        dot(lx[ch][2], y4x4[ir1][...][2]) +
        dot(lx[ch][3], y4x4[ir1][...][3]);
}
```

---

## 附：关键文件索引

| 层级 | 文件 | 职责 |
|------|------|------|
| CLI | `tools/cli/cli.cpp` | 用户交互、任务提交 |
| Server | `tools/server/server-context.cpp` | Batch 调度、调用 `llama_decode` |
| API | `src/llama-context.cpp` | `llama_decode()` → `decode()` → `process_ubatch()` |
| 模型 | `src/llama-model.cpp` | `build_graph()` 架构分发 |
| 图定义 | `src/models/llama.cpp` | Llama Transformer 层的图结构 |
| 图工具 | `src/llama-graph.cpp` | `build_lora_mm()` / `build_ffn()` / `build_attn()` |
| ggml 节点 | `ggml/src/ggml.c` | `ggml_mul_mat()` 创建图节点 |
| 调度器 | `ggml/src/ggml-backend.cpp` | 图切分、后端分发 |
| CPU 计算 | `ggml/src/ggml-cpu/ggml-cpu.c` | `ggml_compute_forward_mul_mat()` |
| CPU kernel | `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_q4_K_q8_K_generic()` |
| CPU SIMD | `ggml/src/ggml-cpu/arch/x86/quants.c` | x86 SIMD 加速 vec_dot |
| CUDA 分发 | `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat()` 路径选择 |
| CUDA MMQ | `ggml/src/ggml-cuda/mmq.cu` | 量化矩阵乘 kernel |
| CUDA 量化 | `ggml/src/ggml-cuda/quantize.cu` | GPU 激活量化 FP32→Q8_1 |
| CUDA dp4a | `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_dp4a()` INT8×INT8 |
| Metal 分发 | `ggml/src/ggml-metal/ggml-metal-ops.cpp` | `ggml_metal_op_mul_mat()` |
| Metal kernel | `ggml/src/ggml-metal/ggml-metal.metal` | 反量化 + FP32 matmul shader |
