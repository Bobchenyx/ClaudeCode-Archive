# Qwen3.5 MoE vs Qwen3 MoE — MoE 架构演进详细报告

> **分析基于**: HuggingFace Transformers 仓库最新代码
> **对比路径**:
> - `transformers/src/transformers/models/qwen3_moe/`
> - `transformers/src/transformers/models/qwen3_5_moe/`
> **分析日期**: 2025-02-25
> **说明**: 本报告聚焦 MoE（Mixture of Experts）架构差异，不涉及多模态（Vision）相关部分。

---

## 一、核心变更总览

Qwen3.5 MoE 相比 Qwen3 MoE，在 MoE 架构上有 **6 项重大变更**：

| # | 变更项 | 简述 |
|---|--------|------|
| 1 | 新增共享专家 (Shared Expert) | 每个 MoE 层增加一个始终激活的共享专家，配合 Sigmoid 门控 |
| 2 | 路由器简化 | 移除 `norm_topk_prob` 开关，强制对 top-k 权重归一化 |
| 3 | 全层 MoE 化 | 移除 `decoder_sparse_step` 和 `mlp_only_layers`，所有层均使用 MoE |
| 4 | 混合注意力机制 | 引入线性注意力 (Gated Delta Net)，与全注意力交替使用 |
| 5 | 专家规模策略转变 | 更多更小的专家 (256×512 vs 128×768) |
| 6 | RMSNorm 1-centered 初始化 | 权重初始化为 0，前向计算 `(1 + weight)` |

---

## 二、配置参数对比

> 对比文件:
> - `qwen3_moe/configuration_qwen3_moe.py` → `Qwen3MoeConfig`
> - `qwen3_5_moe/configuration_qwen3_5_moe.py` → `Qwen3_5MoeTextConfig`

### 2.1 MoE 核心参数

| 参数 | Qwen3 MoE | Qwen3.5 MoE | 变化 |
|------|-----------|-------------|------|
| `num_experts` | 128 | **256** | 专家数量翻倍 |
| `moe_intermediate_size` | 768 | **512** | 单个专家更小 |
| `num_experts_per_tok` | 8 | 8 | 不变 |
| `shared_expert_intermediate_size` | **不存在** | **512** | 新增共享专家 |
| `norm_topk_prob` | `False` (可配置) | **参数已移除** | 强制归一化 |
| `decoder_sparse_step` | 1 (可配置) | **参数已移除** | 全层 MoE |
| `mlp_only_layers` | `[]` (可配置) | **参数已移除** | 不再有 Dense 层 |
| `intermediate_size` | 6144 | **参数已移除** | 无 Dense MLP 故无需 |
| `output_router_logits` | `False` | `False` | 不变 |
| `router_aux_loss_coef` | 0.001 | 0.001 | 不变 |

### 2.2 注意力相关参数

| 参数 | Qwen3 MoE | Qwen3.5 MoE | 变化 |
|------|-----------|-------------|------|
| `head_dim` | 128 (默认) | **256** | 头维度翻倍 |
| `num_attention_heads` | 32 | **16** | 注意力头数减半 |
| `num_key_value_heads` | 4 | **2** | KV 头数减半 |
| `use_sliding_window` | 支持 | **移除** | 不再使用滑动窗口 |
| `sliding_window` | 4096 | **移除** | — |
| `layer_types` | **不存在** | **新增** | `full_attention` / `linear_attention` 交替 |
| `linear_conv_kernel_dim` | **不存在** | **4** | 线性注意力卷积核 |
| `linear_key_head_dim` | **不存在** | **128** | 线性注意力 key 维度 |
| `linear_value_head_dim` | **不存在** | **128** | 线性注意力 value 维度 |
| `linear_num_key_heads` | **不存在** | **16** | 线性注意力 key 头数 |
| `linear_num_value_heads` | **不存在** | **32** | 线性注意力 value 头数 |

### 2.3 其他参数

| 参数 | Qwen3 MoE | Qwen3.5 MoE | 变化 |
|------|-----------|-------------|------|
| `vocab_size` | 151,936 | **248,320** | 词表大幅扩大 |
| `num_hidden_layers` | 24 | **40** | 层数增加 |
| `rms_norm_eps` | 1e-6 | 1e-6 | 不变 |
| `hidden_size` | 2048 | 2048 | 不变 |

---

## 三、核心架构变更详解

### 3.1 变更一：新增共享专家 (Shared Expert) + Sigmoid 门控

这是 Qwen3.5 MoE 最重要的架构变更之一。

#### Qwen3 MoE 的 SparseMoeBlock（无共享专家）

```
输入 ──→ Router (TopK=8 from 128) ──→ 路由专家计算 ──→ 输出
```

源码 (`qwen3_moe/modular_qwen3_moe.py:63-74`):

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)       # 128 个路由专家
        self.gate = Qwen3MoeTopKRouter(config)        # Top-8 路由器

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```

#### Qwen3.5 MoE 的 SparseMoeBlock（含共享专家）

```
输入 ─┬──→ Router (TopK=8 from 256) ──→ 路由专家计算 ────────┐
      │                                                        ├──→ 相加 ──→ 输出
      └──→ 共享专家 MLP ──→ Sigmoid(门控线性层) × 共享输出 ───┘
```

源码 (`qwen3_5_moe/modeling_qwen3_5_moe.py:870-889`):

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)       # Top-8 路由器
        self.experts = Qwen3_5MoeExperts(config)        # 256 个路由专家
        # ★ 新增：共享专家
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size  # 512
        )
        # ★ 新增：共享专家的 Sigmoid 门控
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 共享专家始终处理所有 token
        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        # 路由专家只处理分配到的 token
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        # ★ Sigmoid 门控调节共享专家贡献
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        )

        # 路由专家输出 + 共享专家输出
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
```

#### 设计意图

- **共享专家** 确保每个 token 至少经过一个专家处理，提供稳定的基线表示，避免路由失败时信息丢失
- **Sigmoid 门控** (`nn.Linear(hidden_size, 1)` + `sigmoid`) 让模型动态学习共享专家的贡献比例（0~1 之间）
- 该设计借鉴自 DeepSeekMoE 的 Shared Expert 机制

---

### 3.2 变更二：路由器 (Router) 简化 — 强制归一化

#### Qwen3 MoE 的 TopKRouter（继承自 `Qwen2MoeTopKRouter`）

```python
class Qwen2MoeTopKRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob   # ← 可配置开关
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # ★ 条件性归一化
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

#### Qwen3.5 MoE 的 TopKRouter

```python
class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        # ★ 移除了 norm_topk_prob 属性
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # ★ 无条件归一化 — 永远执行
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

#### 差异总结

| 特征 | Qwen3 MoE | Qwen3.5 MoE |
|------|-----------|-------------|
| `norm_topk_prob` 属性 | 有，从 config 读取 | 移除 |
| 归一化行为 | 条件执行 (`if self.norm_topk_prob`) | **无条件执行** |
| 默认配置 | `norm_topk_prob=False`（不归一化） | 永远归一化 |

**设计意图**: 简化路由逻辑，确保 top-k 路由权重之和始终为 1，使专家输出的缩放更加一致和稳定。

---

### 3.3 变更三：全层 MoE 化 — 移除 Dense MLP 层选项

#### Qwen3 MoE 的 DecoderLayer（有 MoE / Dense 分支）

源码 (`qwen2_moe/modeling_qwen2_moe.py`，`Qwen3MoeDecoderLayer` 继承自此):

```python
class Qwen2MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen2MoeAttention(config, layer_idx)

        # ★ 根据配置决定使用 MoE 还是 Dense MLP
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(config)   # MoE 层
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)  # Dense 层

        self.input_layernorm = ...
        self.post_attention_layernorm = ...
```

这意味着 Qwen3 MoE 支持三种灵活配置：
- `decoder_sparse_step=1`: 所有层都是 MoE
- `decoder_sparse_step=2`: 每隔一层是 MoE，交替 Dense
- `mlp_only_layers=[0,1,2]`: 指定某些层强制使用 Dense MLP

#### Qwen3.5 MoE 的 DecoderLayer（全部 MoE）

源码 (`qwen3_5_moe/modeling_qwen3_5_moe.py:912-924`):

```python
class Qwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)

        # ★ 所有层都使用 SparseMoeBlock，不再有条件判断
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)

        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**设计意图**: 配合共享专家的引入，每一层都使用 MoE（路由专家 + 共享专家），架构更统一。共享专家本身就承担了之前 Dense MLP 的角色。

---

### 3.4 变更四：混合注意力机制 — 引入线性注意力 (Gated Delta Net)

这是 Qwen3.5 MoE 在注意力层面最重要的结构性变革。

#### Qwen3 MoE: 纯全注意力

所有层使用标准 Multi-Head Attention（可选 Sliding Window Attention）：

```python
# Qwen3 MoE — 所有层统一使用 self_attn
class Qwen3MoeDecoderLayer:
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen3MoeAttention(config, layer_idx)  # 全注意力
        ...
```

#### Qwen3.5 MoE: 全注意力 + 线性注意力交替

通过 `layer_types` 配置，按层交替使用两种注意力：

```python
# 默认 layer_types 生成规则 (configuration_qwen3_5_moe.py:198-203)
# full_attention_interval = 4 → 每 4 层一个全注意力层
# 40 层的模式为: [linear, linear, linear, FULL, linear, linear, linear, FULL, ...]

if self.layer_types is None:
    interval_pattern = kwargs.get("full_attention_interval", 4)
    self.layer_types = [
        "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]
```

对于默认的 40 层模型，layer_types 为：

```
Layer  0: linear_attention    Layer 20: linear_attention
Layer  1: linear_attention    Layer 21: linear_attention
Layer  2: linear_attention    Layer 22: linear_attention
Layer  3: full_attention  ★   Layer 23: full_attention  ★
Layer  4: linear_attention    Layer 24: linear_attention
Layer  5: linear_attention    Layer 25: linear_attention
Layer  6: linear_attention    Layer 26: linear_attention
Layer  7: full_attention  ★   Layer 27: full_attention  ★
...                           ...
```

**全注意力层 (10/40 = 25%)**: 使用 `Qwen3_5MoeAttention`，标准 QKV 多头注意力，复杂度 O(n²)
**线性注意力层 (30/40 = 75%)**: 使用 `Qwen3_5MoeGatedDeltaNet`，基于 Gated Delta Net 的线性注意力，复杂度 O(n)

#### DecoderLayer forward 中的分支

```python
def forward(self, hidden_states, position_embeddings, attention_mask, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # ★ Token Mixer 分支
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Feed-Forward (MoE)
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states
```

#### 混合 Cache 机制

由于两种注意力的缓存需求不同，Qwen3.5 MoE 引入了混合 Cache：

```python
class Qwen3_5MoeDynamicCache:
    """同时管理全注意力的 KV Cache 和线性注意力的 recurrent state"""
    def __init__(self, config):
        self.layer_types = config.layer_types
        # 全注意力层: 标准 KV Cache
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        # 线性注意力层: Conv 状态 + 循环状态
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
```

**设计意图**:
- 线性注意力 O(n) 复杂度大幅降低长序列的计算量和内存开销
- 75% 的层用线性注意力 + 25% 的层保留全注意力，兼顾效率与全局建模能力
- Gated Delta Net 是近期在线性注意力领域表现优异的架构，具备高效的递推推理特性

---

### 3.5 变更五：专家规模策略 — 更多更小的专家

#### 激活参数量对比

| 指标 | Qwen3 MoE | Qwen3.5 MoE |
|------|-----------|-------------|
| 路由专家总数 | 128 | **256** |
| 共享专家 | 0 | **1** |
| 单个路由专家中间维度 | 768 | **512** |
| 共享专家中间维度 | — | 512 |
| 每 token 激活路由专家数 | 8 | 8 |
| 每 token 路由激活参数量 | 8 × 768 × 2 × hidden = **大** | 8 × 512 × 2 × hidden = **小** |
| 每 token 共享激活参数量 | 0 | 512 × 2 × hidden |

> 注: 每个专家内部包含 gate_proj + up_proj (合并为 gate_up_proj) 和 down_proj，
> 参数量 ≈ 2 × intermediate_size × hidden_size + hidden_size × intermediate_size = 3 × intermediate_size × hidden_size

**单个路由专家参数量**:
- Qwen3 MoE: 3 × 768 × 2048 = **4,718,592** (~4.7M)
- Qwen3.5 MoE: 3 × 512 × 2048 = **3,145,728** (~3.1M)

**每层路由专家总参数量**:
- Qwen3 MoE: 128 × 4.7M = **603M**
- Qwen3.5 MoE: 256 × 3.1M = **805M** (+ 共享专家 3.1M)

**每 token 激活的 FFN 参数量**:
- Qwen3 MoE: 8 × 4.7M = **37.7M**
- Qwen3.5 MoE: 8 × 3.1M + 3.1M (共享) = **28.0M**

**设计意图**:
- 更多更小的专家 → 更细粒度的专业化分工，每个专家可以专注于更窄的知识领域
- 总参数量增加但激活参数量降低 → 更高的参数效率
- 共享专家兜底通用知识 → 路由专家可以更放心地特化

---

### 3.6 变更六：RMSNorm — 1-Centered 初始化

#### Qwen3 MoE（继承自 `LlamaRMSNorm`）

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(dim))   # ← 初始化为 1
        self.eps = eps

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).to(x.dtype)      # ← 直接乘 weight
```

#### Qwen3.5 MoE（`Qwen3_5MoeRMSNorm`）

```python
class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))   # ★ 初始化为 0
        self.eps = eps

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # ★ 使用 (1 + weight)
        return output.type_as(x)
```

#### 差异对比

| 特性 | Qwen3 MoE | Qwen3.5 MoE |
|------|-----------|-------------|
| weight 初始化 | `torch.ones(dim)` | `torch.zeros(dim)` |
| forward 计算 | `norm(x) * weight` | `norm(x) * (1.0 + weight)` |
| 初始等效行为 | 恒等缩放 (1×) | 恒等缩放 (1+0=1×) |
| 梯度特性 | weight 在 1 附近波动 | weight 在 0 附近波动 |

**设计意图**: 两种写法在初始化时数学等价，但 `(1 + weight)` 形式让参数围绕 0 学习（类似 residual learning 的思想），在优化器中更稳定。这是近期大模型训练中越来越流行的做法。

---

## 四、继承体系对比

### 4.1 Qwen3 MoE 的继承链

```
                        Mixtral (基础 MoE 框架)
                            │
                        Qwen2MoE (加入共享专家框架、sparse step 等)
                            │
                        Qwen3MoE (换用 Qwen3 Attention)
```

具体类继承关系：

```
Qwen3MoeAttention     ← Qwen3Attention (加入 QKNorm)
Qwen3MoeMLP           ← Qwen2MoeMLP
Qwen3MoeExperts        ← Qwen2MoeExperts
Qwen3MoeTopKRouter     ← Qwen2MoeTopKRouter
Qwen3MoeSparseMoeBlock   (独立实现，无共享专家)
Qwen3MoeDecoderLayer   ← Qwen2MoeDecoderLayer (有 MoE/Dense 分支)
Qwen3MoeRMSNorm        ← LlamaRMSNorm
Qwen3MoeModel          ← MixtralModel
Qwen3MoeForCausalLM    ← MixtralForCausalLM
```

### 4.2 Qwen3.5 MoE 的继承链

```
                        Mixtral (基础 MoE 框架)
                            │
                        Qwen2MoE (加入共享专家框架)
                            │
                        Qwen3MoE (换用 Qwen3 Attention)
                            │
                        Qwen3Next (★ 新引入的中间层)
                            │    ├─ 混合注意力 (full + linear)
                            │    ├─ 全层 MoE 化
                            │    ├─ 新 RMSNorm
                            │    └─ 混合 Cache
                            │
                    ┌───────┴───────┐
                Qwen3_5         Qwen3VLMoe
                    │               │
                Qwen3_5Moe ◄────────┘ (路由器来自 VLMoe)
```

具体类继承关系：

```
Qwen3_5MoeAttention       ← Qwen3NextAttention
Qwen3_5MoeMLP             ← Qwen3_5MLP (来自 qwen3_5 非 MoE 版)
Qwen3_5MoeExperts          ← Qwen3NextExperts ← Qwen2MoeExperts
Qwen3_5MoeTopKRouter       ← Qwen3VLMoeTextTopKRouter (★ 强制归一化路由)
Qwen3_5MoeSparseMoeBlock   ← Qwen3NextSparseMoeBlock (含共享专家)
Qwen3_5MoeGatedDeltaNet    ← Qwen3_5GatedDeltaNet (★ 线性注意力)
Qwen3_5MoeDecoderLayer     ← Qwen3NextDecoderLayer (混合注意力)
Qwen3_5MoeRMSNorm          ← Qwen3NextRMSNorm (1-centered)
Qwen3_5MoeDynamicCache     ← Qwen3NextDynamicCache (混合 Cache)
Qwen3_5MoePreTrainedModel  ← Qwen3NextPreTrainedModel
```

---

## 五、TP (Tensor Parallelism) 计划对比

Qwen3.5 MoE 的 TP 计划反映了共享专家的引入：

### Qwen3 MoE

```python
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    # ★ Dense MLP 层的 TP 计划
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```

### Qwen3.5 MoE

```python
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    # ★ 共享专家的 TP 计划（替代了 Dense MLP 的 TP 计划）
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
}
```

差异：Qwen3 MoE 的 `layers.*.mlp.{gate,up,down}_proj` 指向非 MoE 层的 Dense MLP；Qwen3.5 MoE 不再有 Dense MLP，取而代之的是 `layers.*.mlp.shared_expert.*` 指向共享专家。

---

## 六、权重初始化对比

### Qwen3 MoE

继承自 `MixtralPreTrainedModel`，使用标准初始化：
- 所有线性层: `normal_(mean=0, std=config.initializer_range)`
- RMSNorm: `ones_(weight)` (torch 默认)

### Qwen3.5 MoE

在 `Qwen3_5MoePreTrainedModel._init_weights` 中定义了更精细的初始化策略：

```python
def _init_weights(self, module):
    if isinstance(module, Qwen3_5MoeGatedDeltaNet):
        init.ones_(module.dt_bias)                    # Delta Net 的时间步偏置初始化为 1
        init.copy_(module.A_log,
            torch.empty_like(module.A_log).uniform_(0, 16).log_()  # A_log 对数均匀分布
        )
    elif isinstance(module, Qwen3_5MoeRMSNorm):
        init.zeros_(module.weight)                     # ★ RMSNorm 权重初始化为 0
    elif isinstance(module, Qwen3_5MoeExperts):
        init.normal_(module.gate_up_proj, mean=0.0, std=config.initializer_range)
        init.normal_(module.down_proj, mean=0.0, std=config.initializer_range)
    elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
        init.normal_(module.gate.weight, mean=0.0, std=config.initializer_range)  # 路由器权重
```

---

## 七、架构图示总结

### Qwen3 MoE 单层结构

```
                    ┌─────────────────────────────┐
                    │       DecoderLayer           │
                    │                              │
Input ──→ LN ──→ Self-Attention (全注意力) ──→ Add ──→ LN ──→ ┌─ MoE Block ─┐ ──→ Add ──→ Output
     (residual)                             (residual)        │  或          │
                                                              │  Dense MLP   │
                                                              └──────────────┘
                    MoE Block (当选择 MoE 时):
                    ┌─────────────────────────────────┐
                    │  Router → Top-8 from 128 experts │
                    │  Experts: 稀疏计算              │
                    │  (无共享专家)                     │
                    └─────────────────────────────────┘
```

### Qwen3.5 MoE 单层结构

```
                    ┌──────────────────────────────────────────────────┐
                    │              DecoderLayer                        │
                    │                                                  │
                    │  ┌─ layer_type 分支 ──────────────────────────┐  │
Input ──→ LN ──→   │  │ "full_attention" → Self-Attention (QKV)   │  │ ──→ Add ──→ LN ──→ MoE Block ──→ Add ──→ Out
     (residual)     │  │ "linear_attention" → GatedDeltaNet (O(n)) │  │      (residual)
                    │  └───────────────────────────────────────────┘  │
                    └──────────────────────────────────────────────────┘

                    MoE Block (所有层):
                    ┌──────────────────────────────────────────────────┐
                    │                                                  │
                    │  ┌─→ Router → Top-8 from 256 → Routed Experts ─┐│
                    │  │                                              ││
             Input ─┤                                          Add ──┤├─→ Output
                    │  │                                              ││
                    │  └─→ Shared Expert → Sigmoid Gate ─────────────┘│
                    │                                                  │
                    └──────────────────────────────────────────────────┘
```

---

## 八、总结对比表

| 维度 | Qwen3 MoE | Qwen3.5 MoE |
|------|-----------|-------------|
| **路由专家数** | 128 | **256** (+100%) |
| **共享专家** | 无 | **有** (Sigmoid 门控) |
| **单专家大小** | intermediate=768 | **intermediate=512** (-33%) |
| **每token激活专家** | 8 路由 | 8 路由 + **1 共享** |
| **路由归一化** | 可选 (`norm_topk_prob`) | **强制归一化** |
| **MoE 层分布** | 可配置 MoE/Dense 混合 | **全部 MoE** |
| **注意力机制** | 全注意力 (+ 可选 SWA) | **混合: 线性注意力 75% + 全注意力 25%** |
| **线性注意力** | 无 | **Gated Delta Net** (O(n) 复杂度) |
| **RMSNorm** | 标准 (weight × norm) | **1-centered ((1+weight) × norm)** |
| **Cache** | 标准 KV Cache | **混合 Cache** (KV + conv/recurrent) |
| **TP 策略** | Dense MLP + Experts | **Shared Expert + Experts** |

### 核心设计哲学

**Qwen3.5 MoE 的架构演进方向是**:

1. **"更多更小"的专家策略** — 256 个小专家取代 128 个大专家，实现更细粒度的知识特化
2. **"共享 + 路由"双轨制** — 共享专家保底通用能力，路由专家专注特化知识
3. **"线性 + 全局"混合注意力** — 75% 线性注意力大幅降本，25% 全注意力保持全局建模
4. **简化配置空间** — 移除多个可选开关 (`norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers`)，减少超参调优负担

这是一种在 **激活参数量不变甚至降低** 的前提下，通过增加总参数量和架构异质性来提升模型能力的路线。
