# Qwen3.5 MoE vs Qwen3 MoE — MoE Architecture Evolution Report

> **Based on**: Latest HuggingFace Transformers repository code
> **Compared paths**:
> - `transformers/src/transformers/models/qwen3_moe/`
> - `transformers/src/transformers/models/qwen3_5_moe/`
> **Date**: 2025-02-25
> **Scope**: This report focuses exclusively on MoE (Mixture of Experts) architecture differences. Multimodal (Vision) components are excluded.

---

## 1. Summary of Key Changes

Qwen3.5 MoE introduces **6 major architectural changes** to the MoE system compared to Qwen3 MoE:

| # | Change | Description |
|---|--------|-------------|
| 1 | Shared Expert Addition | Each MoE layer gains an always-active shared expert with sigmoid gating |
| 2 | Router Simplification | Removed `norm_topk_prob` toggle; top-k weights are always renormalized |
| 3 | All-MoE Layers | Removed `decoder_sparse_step` and `mlp_only_layers`; every layer uses MoE |
| 4 | Hybrid Attention | Introduced linear attention (Gated Delta Net) alternating with full attention |
| 5 | Expert Scaling Strategy | More and smaller experts (256x512 vs 128x768) |
| 6 | 1-Centered RMSNorm | Weight initialized to zero, forward computes `(1 + weight)` |

---

## 2. Configuration Parameter Comparison

> Compared files:
> - `qwen3_moe/configuration_qwen3_moe.py` -> `Qwen3MoeConfig`
> - `qwen3_5_moe/configuration_qwen3_5_moe.py` -> `Qwen3_5MoeTextConfig`

### 2.1 Core MoE Parameters

| Parameter | Qwen3 MoE | Qwen3.5 MoE | Change |
|-----------|-----------|-------------|--------|
| `num_experts` | 128 | **256** | Doubled |
| `moe_intermediate_size` | 768 | **512** | Smaller per-expert |
| `num_experts_per_tok` | 8 | 8 | Unchanged |
| `shared_expert_intermediate_size` | **N/A** | **512** | New shared expert |
| `norm_topk_prob` | `False` (configurable) | **Removed** | Always normalize |
| `decoder_sparse_step` | 1 (configurable) | **Removed** | All layers are MoE |
| `mlp_only_layers` | `[]` (configurable) | **Removed** | No dense-only layers |
| `intermediate_size` | 6144 | **Removed** | No dense MLP needed |
| `output_router_logits` | `False` | `False` | Unchanged |
| `router_aux_loss_coef` | 0.001 | 0.001 | Unchanged |

### 2.2 Attention Parameters

| Parameter | Qwen3 MoE | Qwen3.5 MoE | Change |
|-----------|-----------|-------------|--------|
| `head_dim` | 128 (default) | **256** | Doubled |
| `num_attention_heads` | 32 | **16** | Halved |
| `num_key_value_heads` | 4 | **2** | Halved |
| `use_sliding_window` | Supported | **Removed** | No longer used |
| `sliding_window` | 4096 | **Removed** | -- |
| `layer_types` | **N/A** | **New** | `full_attention` / `linear_attention` alternating |
| `linear_conv_kernel_dim` | **N/A** | **4** | Linear attention conv kernel |
| `linear_key_head_dim` | **N/A** | **128** | Linear attention key dim |
| `linear_value_head_dim` | **N/A** | **128** | Linear attention value dim |
| `linear_num_key_heads` | **N/A** | **16** | Linear attention key heads |
| `linear_num_value_heads` | **N/A** | **32** | Linear attention value heads |

### 2.3 Other Parameters

| Parameter | Qwen3 MoE | Qwen3.5 MoE | Change |
|-----------|-----------|-------------|--------|
| `vocab_size` | 151,936 | **248,320** | Significantly expanded |
| `num_hidden_layers` | 24 | **40** | More layers |
| `rms_norm_eps` | 1e-6 | 1e-6 | Unchanged |
| `hidden_size` | 2048 | 2048 | Unchanged |

---

## 3. Detailed Architecture Changes

### 3.1 Change 1: Shared Expert with Sigmoid Gating

This is one of the most significant architectural changes in Qwen3.5 MoE.

#### Qwen3 MoE SparseMoeBlock (No Shared Expert)

```
Input --> Router (TopK=8 from 128) --> Routed Experts --> Output
```

Source (`qwen3_moe/modular_qwen3_moe.py:63-74`):

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)       # 128 routed experts
        self.gate = Qwen3MoeTopKRouter(config)        # Top-8 router

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
```

#### Qwen3.5 MoE SparseMoeBlock (With Shared Expert)

```
Input -+---> Router (TopK=8 from 256) --> Routed Experts --------+
       |                                                          +--> Add --> Output
       +---> Shared Expert MLP --> Sigmoid(gate_linear) x out ---+
```

Source (`qwen3_5_moe/modeling_qwen3_5_moe.py:870-889`):

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)       # Top-8 router
        self.experts = Qwen3_5MoeExperts(config)        # 256 routed experts
        # NEW: Shared expert
        self.shared_expert = Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size  # 512
        )
        # NEW: Sigmoid gate for shared expert
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # Shared expert always processes all tokens
        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        # Routed experts only process assigned tokens
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        # Sigmoid gate modulates shared expert contribution
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        )

        # Combine routed + shared expert outputs
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
```

#### Design Rationale

- The **shared expert** ensures every token receives processing from at least one expert, providing a stable baseline representation and preventing information loss from routing failures.
- The **sigmoid gate** (`nn.Linear(hidden_size, 1)` + `sigmoid`) allows the model to dynamically learn the contribution ratio of the shared expert (between 0 and 1).
- This design is inspired by the Shared Expert mechanism from DeepSeekMoE.

---

### 3.2 Change 2: Router Simplification -- Mandatory Normalization

#### Qwen3 MoE TopKRouter (inherits from `Qwen2MoeTopKRouter`)

```python
class Qwen2MoeTopKRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob   # <-- Configurable toggle
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # Conditional normalization
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

#### Qwen3.5 MoE TopKRouter

```python
class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        # norm_topk_prob attribute REMOVED
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # UNCONDITIONAL normalization -- always executed
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

#### Comparison

| Feature | Qwen3 MoE | Qwen3.5 MoE |
|---------|-----------|-------------|
| `norm_topk_prob` attribute | Present, read from config | Removed |
| Normalization behavior | Conditional (`if self.norm_topk_prob`) | **Always applied** |
| Default behavior | `norm_topk_prob=False` (no normalization) | Always normalize |

**Design Rationale**: Simplifies routing logic by ensuring top-k routing weights always sum to 1, making expert output scaling more consistent and stable.

---

### 3.3 Change 3: All-MoE Layers -- Dense MLP Option Removed

#### Qwen3 MoE DecoderLayer (MoE / Dense Branching)

Source (`qwen2_moe/modeling_qwen2_moe.py`, base class for `Qwen3MoeDecoderLayer`):

```python
class Qwen2MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen2MoeAttention(config, layer_idx)

        # Decides between MoE and Dense MLP based on config
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(config)   # MoE layer
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)  # Dense layer

        self.input_layernorm = ...
        self.post_attention_layernorm = ...
```

This allowed Qwen3 MoE three flexible configuration patterns:
- `decoder_sparse_step=1`: All layers use MoE
- `decoder_sparse_step=2`: Alternating MoE and Dense layers
- `mlp_only_layers=[0,1,2]`: Force specific layers to use Dense MLP

#### Qwen3.5 MoE DecoderLayer (All MoE)

Source (`qwen3_5_moe/modeling_qwen3_5_moe.py:912-924`):

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

        # ALL layers use SparseMoeBlock -- no conditional branching
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)

        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**Design Rationale**: With the introduction of shared experts, every layer uses MoE (routed experts + shared expert), creating a more uniform architecture. The shared expert effectively takes on the role previously filled by dense MLP layers.

---

### 3.4 Change 4: Hybrid Attention -- Linear Attention (Gated Delta Net)

This is the most important structural change at the attention level.

#### Qwen3 MoE: Full Attention Only

All layers use standard Multi-Head Attention (with optional Sliding Window Attention):

```python
# Qwen3 MoE -- all layers uniformly use self_attn
class Qwen3MoeDecoderLayer:
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen3MoeAttention(config, layer_idx)  # Full attention
        ...
```

#### Qwen3.5 MoE: Full Attention + Linear Attention Alternating

The `layer_types` configuration enables per-layer attention type selection:

```python
# Default layer_types generation (configuration_qwen3_5_moe.py:198-203)
# full_attention_interval = 4 --> one full attention layer every 4 layers
# 40-layer pattern: [linear, linear, linear, FULL, linear, linear, linear, FULL, ...]

if self.layer_types is None:
    interval_pattern = kwargs.get("full_attention_interval", 4)
    self.layer_types = [
        "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]
```

For the default 40-layer model, the `layer_types` pattern is:

```
Layer  0: linear_attention    Layer 20: linear_attention
Layer  1: linear_attention    Layer 21: linear_attention
Layer  2: linear_attention    Layer 22: linear_attention
Layer  3: full_attention  *   Layer 23: full_attention  *
Layer  4: linear_attention    Layer 24: linear_attention
Layer  5: linear_attention    Layer 25: linear_attention
Layer  6: linear_attention    Layer 26: linear_attention
Layer  7: full_attention  *   Layer 27: full_attention  *
...                           ...
```

**Full attention layers (10/40 = 25%)**: Use `Qwen3_5MoeAttention` -- standard QKV multi-head attention with O(n^2) complexity.
**Linear attention layers (30/40 = 75%)**: Use `Qwen3_5MoeGatedDeltaNet` -- Gated Delta Net linear attention with O(n) complexity.

#### DecoderLayer Forward Branching

```python
def forward(self, hidden_states, position_embeddings, attention_mask, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Token Mixer branch
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

#### Hybrid Cache Mechanism

Since the two attention types have different caching requirements, Qwen3.5 MoE introduces a hybrid cache:

```python
class Qwen3_5MoeDynamicCache:
    """Manages both full attention KV Cache and linear attention recurrent state"""
    def __init__(self, config):
        self.layer_types = config.layer_types
        # Full attention layers: standard KV Cache
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        # Linear attention layers: Conv state + Recurrent state
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
```

#### Design Rationale

- Linear attention with O(n) complexity dramatically reduces computation and memory overhead for long sequences.
- 75% linear attention + 25% full attention balances efficiency with global modeling capability.
- Gated Delta Net is a recent high-performing architecture in the linear attention space, featuring efficient recurrent inference properties.

---

### 3.5 Change 5: Expert Scaling Strategy -- More and Smaller Experts

#### Activated Parameter Comparison

| Metric | Qwen3 MoE | Qwen3.5 MoE |
|--------|-----------|-------------|
| Total routed experts | 128 | **256** |
| Shared experts | 0 | **1** |
| Per-expert intermediate dim | 768 | **512** |
| Shared expert intermediate dim | -- | 512 |
| Activated routed experts per token | 8 | 8 |

> Note: Each expert contains gate_proj + up_proj (merged as gate_up_proj) and down_proj.
> Parameters per expert ~ 2 x intermediate_size x hidden_size + hidden_size x intermediate_size = 3 x intermediate_size x hidden_size

**Parameters per routed expert**:
- Qwen3 MoE: 3 x 768 x 2048 = **4,718,592** (~4.7M)
- Qwen3.5 MoE: 3 x 512 x 2048 = **3,145,728** (~3.1M)

**Total routed expert parameters per layer**:
- Qwen3 MoE: 128 x 4.7M = **603M**
- Qwen3.5 MoE: 256 x 3.1M = **805M** (+ 3.1M for shared expert)

**Activated FFN parameters per token**:
- Qwen3 MoE: 8 x 4.7M = **37.7M**
- Qwen3.5 MoE: 8 x 3.1M + 3.1M (shared) = **28.0M**

#### Design Rationale

- **More and smaller experts** enable finer-grained specialization -- each expert can focus on a narrower knowledge domain.
- **Higher total parameters but lower activated parameters** lead to greater parameter efficiency.
- **Shared expert handles general knowledge**, allowing routed experts to specialize more aggressively.

---

### 3.6 Change 6: RMSNorm -- 1-Centered Initialization

#### Qwen3 MoE (inherits from `LlamaRMSNorm`)

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(dim))   # <-- Initialized to 1
        self.eps = eps

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).to(x.dtype)      # <-- Direct multiplication
```

#### Qwen3.5 MoE (`Qwen3_5MoeRMSNorm`)

```python
class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))   # Initialized to 0
        self.eps = eps

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # Uses (1 + weight)
        return output.type_as(x)
```

#### Comparison

| Feature | Qwen3 MoE | Qwen3.5 MoE |
|---------|-----------|-------------|
| Weight initialization | `torch.ones(dim)` | `torch.zeros(dim)` |
| Forward computation | `norm(x) * weight` | `norm(x) * (1.0 + weight)` |
| Initial effective behavior | Identity scaling (1x) | Identity scaling (1+0=1x) |
| Gradient characteristics | Weight fluctuates around 1 | Weight fluctuates around 0 |

**Design Rationale**: Both formulations are mathematically equivalent at initialization, but the `(1 + weight)` form keeps parameters centered around 0 during learning (analogous to residual learning), which provides better stability with optimizers. This is an increasingly popular technique in large model training.

---

## 4. Inheritance Hierarchy Comparison

### 4.1 Qwen3 MoE Inheritance Chain

```
                        Mixtral (Base MoE framework)
                            |
                        Qwen2MoE (Adds shared expert framework, sparse step, etc.)
                            |
                        Qwen3MoE (Switches to Qwen3 Attention with QKNorm)
```

Class-level inheritance:

```
Qwen3MoeAttention      <-- Qwen3Attention (adds QKNorm)
Qwen3MoeMLP            <-- Qwen2MoeMLP
Qwen3MoeExperts         <-- Qwen2MoeExperts
Qwen3MoeTopKRouter      <-- Qwen2MoeTopKRouter
Qwen3MoeSparseMoeBlock     (standalone implementation, no shared expert)
Qwen3MoeDecoderLayer    <-- Qwen2MoeDecoderLayer (MoE/Dense branching)
Qwen3MoeRMSNorm         <-- LlamaRMSNorm
Qwen3MoeModel           <-- MixtralModel
Qwen3MoeForCausalLM     <-- MixtralForCausalLM
```

### 4.2 Qwen3.5 MoE Inheritance Chain

```
                        Mixtral (Base MoE framework)
                            |
                        Qwen2MoE (Adds shared expert framework)
                            |
                        Qwen3MoE (Switches to Qwen3 Attention)
                            |
                        Qwen3Next (* Newly introduced intermediate layer)
                            |    +-- Hybrid attention (full + linear)
                            |    +-- All-MoE layers
                            |    +-- New RMSNorm
                            |    +-- Hybrid Cache
                            |
                    +-------+-------+
                Qwen3_5         Qwen3VLMoe
                    |               |
                Qwen3_5Moe <--------+ (Router from VLMoe)
```

Class-level inheritance:

```
Qwen3_5MoeAttention        <-- Qwen3NextAttention
Qwen3_5MoeMLP              <-- Qwen3_5MLP (from non-MoE qwen3_5)
Qwen3_5MoeExperts           <-- Qwen3NextExperts <-- Qwen2MoeExperts
Qwen3_5MoeTopKRouter        <-- Qwen3VLMoeTextTopKRouter (mandatory normalization)
Qwen3_5MoeSparseMoeBlock    <-- Qwen3NextSparseMoeBlock (with shared expert)
Qwen3_5MoeGatedDeltaNet     <-- Qwen3_5GatedDeltaNet (linear attention)
Qwen3_5MoeDecoderLayer      <-- Qwen3NextDecoderLayer (hybrid attention)
Qwen3_5MoeRMSNorm           <-- Qwen3NextRMSNorm (1-centered)
Qwen3_5MoeDynamicCache      <-- Qwen3NextDynamicCache (hybrid cache)
Qwen3_5MoePreTrainedModel   <-- Qwen3NextPreTrainedModel
```

---

## 5. Tensor Parallelism (TP) Plan Comparison

The Qwen3.5 MoE TP plan reflects the introduction of shared experts:

### Qwen3 MoE

```python
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    # TP plan for Dense MLP layers
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
    # TP plan for Shared Expert (replaces Dense MLP TP plan)
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
}
```

**Key Difference**: In Qwen3 MoE, `layers.*.mlp.{gate,up,down}_proj` targets the Dense MLP in non-MoE layers. In Qwen3.5 MoE, there are no Dense MLP layers; instead, `layers.*.mlp.shared_expert.*` targets the shared expert within each MoE block.

---

## 6. Weight Initialization Comparison

### Qwen3 MoE

Inherits from `MixtralPreTrainedModel` with standard initialization:
- All linear layers: `normal_(mean=0, std=config.initializer_range)`
- RMSNorm: `ones_(weight)` (PyTorch default)

### Qwen3.5 MoE

Defines a more granular initialization strategy in `Qwen3_5MoePreTrainedModel._init_weights`:

```python
def _init_weights(self, module):
    if isinstance(module, Qwen3_5MoeGatedDeltaNet):
        init.ones_(module.dt_bias)                    # Delta Net timestep bias = 1
        init.copy_(module.A_log,
            torch.empty_like(module.A_log).uniform_(0, 16).log_()  # Log-uniform for A_log
        )
    elif isinstance(module, Qwen3_5MoeRMSNorm):
        init.zeros_(module.weight)                     # RMSNorm weight = 0
    elif isinstance(module, Qwen3_5MoeExperts):
        init.normal_(module.gate_up_proj, mean=0.0, std=config.initializer_range)
        init.normal_(module.down_proj, mean=0.0, std=config.initializer_range)
    elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
        init.normal_(module.gate.weight, mean=0.0, std=config.initializer_range)  # Router weights
```

---

## 7. Architecture Diagrams

### Qwen3 MoE -- Single Layer Structure

```
                    +-----------------------------+
                    |       DecoderLayer           |
                    |                              |
Input --> LN --> Self-Attention (Full) --> Add --> LN --> +- MoE Block -+ --> Add --> Output
     (residual)                        (residual)        |  or         |
                                                         |  Dense MLP  |
                                                         +-------------+

                    MoE Block (when selected):
                    +---------------------------------+
                    |  Router -> Top-8 from 128       |
                    |  Experts: sparse computation    |
                    |  (no shared expert)              |
                    +---------------------------------+
```

### Qwen3.5 MoE -- Single Layer Structure

```
                    +--------------------------------------------------+
                    |              DecoderLayer                         |
                    |                                                   |
                    |  +- layer_type branch --------------------------+ |
Input --> LN -->    |  | "full_attention"   -> Self-Attention (QKV)   | | --> Add --> LN --> MoE Block --> Add --> Out
     (residual)     |  | "linear_attention" -> GatedDeltaNet (O(n))   | |      (residual)
                    |  +----------------------------------------------+ |
                    +--------------------------------------------------+

                    MoE Block (all layers):
                    +--------------------------------------------------+
                    |                                                   |
                    |  +--> Router -> Top-8 from 256 -> Routed Exp. --+|
                    |  |                                               ||
             Input -+                                           Add --++--> Output
                    |  |                                               ||
                    |  +--> Shared Expert -> Sigmoid Gate -------------+|
                    |                                                   |
                    +--------------------------------------------------+
```

---

## 8. Summary Comparison Table

| Dimension | Qwen3 MoE | Qwen3.5 MoE |
|-----------|-----------|-------------|
| **Routed experts** | 128 | **256** (+100%) |
| **Shared expert** | None | **Yes** (sigmoid-gated) |
| **Per-expert size** | intermediate=768 | **intermediate=512** (-33%) |
| **Activated experts/token** | 8 routed | 8 routed + **1 shared** |
| **Router normalization** | Optional (`norm_topk_prob`) | **Always normalize** |
| **MoE layer distribution** | Configurable MoE/Dense mix | **All MoE** |
| **Attention mechanism** | Full attention (+ optional SWA) | **Hybrid: 75% linear + 25% full** |
| **Linear attention** | None | **Gated Delta Net** (O(n) complexity) |
| **RMSNorm** | Standard (weight x norm) | **1-centered ((1+weight) x norm)** |
| **Cache** | Standard KV Cache | **Hybrid Cache** (KV + conv/recurrent) |
| **TP strategy** | Dense MLP + Experts | **Shared Expert + Experts** |

---

## 9. Core Design Philosophy

The architectural evolution of Qwen3.5 MoE follows four key principles:

1. **"More and Smaller" Expert Strategy** -- 256 small experts replace 128 large ones, enabling finer-grained knowledge specialization across experts.

2. **"Shared + Routed" Dual-Track System** -- The shared expert provides a universal capability baseline, freeing routed experts to specialize more aggressively in narrow domains.

3. **"Linear + Full" Hybrid Attention** -- 75% linear attention (Gated Delta Net) dramatically reduces computational cost while 25% full attention layers preserve global modeling capability.

4. **Simplified Configuration Space** -- Removal of multiple optional toggles (`norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers`) reduces the hyperparameter tuning burden and enforces architectural consistency.

**In essence, Qwen3.5 MoE achieves higher model capacity through increased total parameters and architectural heterogeneity, while maintaining or even reducing the activated parameter count per token** -- a design philosophy that maximizes parameter efficiency.
