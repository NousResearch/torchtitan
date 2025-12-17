# Comprehensive Ablation Plan: 80B_A3B Qwen3-Next MoE

## Executive Summary

This document provides a comprehensive ablation plan for the **80B_A3B Qwen3-Next MoE** architecture, covering every architectural component identified through code analysis. The plan is designed to systematically explore design choices to optimize model quality and training efficiency.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Model Configuration Summary](#2-model-configuration-summary)
3. [Ablation Categories](#3-ablation-categories)
   - 3.1 [Attention Mechanism](#31-attention-mechanism)
   - 3.2 [Hybrid Attention Pattern](#32-hybrid-attention-pattern)
   - 3.3 [MoE Routing](#33-moe-routing)
   - 3.4 [Expert Architecture](#34-expert-architecture)
   - 3.5 [Shared Experts](#35-shared-experts)
   - 3.6 [Load Balancing](#36-load-balancing)
   - 3.7 [Normalization](#37-normalization)
   - 3.8 [Positional Encoding](#38-positional-encoding)
   - 3.9 [Initialization](#39-initialization)
   - 3.10 [FFN Architecture](#310-ffn-architecture)
4. [Ablation Configurations](#4-ablation-configurations)
5. [Metrics & Analysis](#5-metrics--analysis)
6. [Priority & Timeline](#6-priority--timeline)
7. [Training Configuration for Ablations](#7-training-configuration-for-ablations)

---

## 1. Architecture Overview

### Source Files Analyzed

| File | Purpose |
|------|---------|
| `model/model.py` | Main model architecture (Qwen3NextModel, Attention, GatedDeltaNet, TransformerBlock) |
| `model/args.py` | Model configuration (Qwen3NextModelArgs) |
| `__init__.py` | Config definitions and TrainSpec |
| `torchtitan/models/moe/moe.py` | MoE implementation (MoE, MoEArgs, TokenChoiceTopKRouter, GroupedExperts) |
| `torchtitan/models/moe/utils.py` | Token permutation utilities |
| `torchtitan/models/moe/kernels.py` | Triton kernels for permutation |
| `infra/parallelize.py` | Parallelism strategies |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Qwen3NextModel                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ tok_embeddings: nn.Embedding(vocab_size=151936, dim=2048)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TransformerBlock × 48                             │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ attention_norm: ZeroCenteredRMSNorm(dim=2048)                  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ attention: Attention OR GatedDeltaNet (hybrid pattern)         │ │   │
│  │  │                                                                │ │   │
│  │  │ FULL ATTENTION (layers 3,7,11,...):                            │ │   │
│  │  │   • wq: Linear(dim, n_heads×head_dim×2) [includes gate]        │ │   │
│  │  │   • wk: Linear(dim, n_kv_heads×head_dim)                       │ │   │
│  │  │   • wv: Linear(dim, n_kv_heads×head_dim)                       │ │   │
│  │  │   • wo: Linear(n_heads×head_dim, dim)                          │ │   │
│  │  │   • q_norm, k_norm: ZeroCenteredRMSNorm(head_dim)              │ │   │
│  │  │   • Gated output: output * sigmoid(gate)                       │ │   │
│  │  │                                                                │ │   │
│  │  │ LINEAR ATTENTION (layers 0,1,2,4,5,6,...):                     │ │   │
│  │  │   • GatedDeltaNet (flash-linear-attention)                     │ │   │
│  │  │   • Causal conv1d for local context                            │ │   │
│  │  │   • Gated delta rule recurrence                                │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                       │   │
│  │                              ▼ (residual)                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ffn_norm: ZeroCenteredRMSNorm(dim=2048)                        │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ MoE Layer (all layers with decoder_sparse_step=1)              │ │   │
│  │  │                                                                │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ router: TokenChoiceTopKRouter                            │  │ │   │
│  │  │  │   • gate: Linear(dim, num_experts=512)                   │  │ │   │
│  │  │  │   • score_func: softmax (normalized)                     │  │ │   │
│  │  │  │   • top_k: 10                                            │  │ │   │
│  │  │  │   • route_norm: True (normalize top-k scores)            │  │ │   │
│  │  │  │   • route_scale: 1.0                                     │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  │                              │                                  │ │   │
│  │  │                              ▼                                  │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ experts: GroupedExperts                                  │  │ │   │
│  │  │  │   • num_experts: 512                                     │  │ │   │
│  │  │  │   • w1: (512, moe_inter_dim=512, dim=2048)               │  │ │   │
│  │  │  │   • w2: (512, dim=2048, moe_inter_dim=512)               │  │ │   │
│  │  │  │   • w3: (512, moe_inter_dim=512, dim=2048)               │  │ │   │
│  │  │  │   • SwiGLU: w2(silu(w1(x)) * w3(x))                      │  │ │   │
│  │  │  │   • use_grouped_mm: True (torch._grouped_mm)             │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  │                              │                                  │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ shared_experts: FeedForward (always active)              │  │ │   │
│  │  │  │   • num_shared_experts: 1                                │  │ │   │
│  │  │  │   • hidden_dim: 5120 (larger than routed experts)        │  │ │   │
│  │  │  │   • shared_gate: True (learnable gate sigmoid)           │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  │                                                                │ │   │
│  │  │  Output: shared_out * shared_gate + Σ(routed_out × scores)     │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ norm: ZeroCenteredRMSNorm(dim=2048)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ output: nn.Linear(dim=2048, vocab_size=151936)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Configuration Summary

### 80B_A3B Default Configuration

```python
Qwen3NextModelArgs(
    # Core dimensions
    dim=2048,                      # Model hidden dimension
    n_layers=48,                   # Number of transformer layers
    n_heads=16,                    # Number of attention heads
    n_kv_heads=2,                  # Number of KV heads (GQA ratio = 8)
    head_dim=256,                  # Dimension per head
    vocab_size=151936,             # Vocabulary size

    # FFN dimensions
    hidden_dim=5120,               # Shared FFN intermediate dim
    hidden_act="silu",             # Activation function

    # MoE Configuration
    moe_enabled=True,
    moe_inter_dim=512,             # Expert FFN intermediate dim (SMALL!)
    moe_args=MoEArgs(
        num_experts=512,           # Total number of experts
        num_shared_experts=1,      # Shared experts (always active)
        top_k=10,                  # Experts selected per token
        score_func="softmax",      # Routing score function
        route_norm=True,           # Normalize routing scores
        route_scale=1.0,           # Scale factor for scores
        score_before_experts=False,# Score AFTER expert forward
        shared_gate=True,          # Learnable gate for shared expert
        load_balance_coeff=1e-3,   # Aux-loss-free LB coefficient
        use_grouped_mm=True,       # Use torch._grouped_mm
    ),

    # Hybrid Attention
    decoder_sparse_step=1,         # MoE every N layers (1 = all layers)
    full_attention_interval=4,     # Full attn every N layers
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "full_attention", ...],  # Pattern: 3 linear, 1 full

    # Linear Attention (GatedDeltaNet) params
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,

    # Positional Encoding
    rope_theta=1000000,            # RoPE base frequency
    partial_rotary_factor=0.25,    # Only 25% of head_dim gets RoPE
    max_seq_len=4096,              # Maximum sequence length

    # Normalization
    norm_eps=1e-6,                 # RMSNorm epsilon

    # Initialization
    depth_init=True,               # Layer-dependent init std

    # Attention
    use_flex_attn=True,            # Use FlexAttention
    attn_mask_type="block_causal", # Block-causal masking
)
```

### Parameter Breakdown

| Component | Per Layer | Total (48L) | Notes |
|-----------|-----------|-------------|-------|
| **Attention (Q+K+V+O)** | 18.9M | 0.91B | Q includes gate projection |
| **Expert FFN** | 1610.6M | 77.31B | 512 experts × 3.1M each |
| **Shared FFN** | 31.5M | 1.51B | 1 shared expert, larger dim |
| **Router** | 1.05M | 0.05B | dim → num_experts projection |
| **LayerNorms** | 0.008M | 0.4M | Zero-centered RMSNorm |
| **Embeddings** | - | 0.62B | 151936 × 2048 × 2 |
| **TOTAL** | - | **80.4B** | - |
| **ACTIVE** | - | **4.6B** | - |
| **SPARSITY** | - | **17.5x** | - |

---

## 3. Ablation Categories

### 3.1 Attention Mechanism

#### 3.1.1 Full Attention Components

**Location**: `model/model.py:127-209` (Attention class)

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `n_heads` | 16 | 8, 16, 32 | More heads = finer attention patterns |
| `n_kv_heads` | 2 | 1, 2, 4, 8, 16 | GQA ratio affects memory/quality |
| `head_dim` | 256 | 64, 128, 256 | Larger head_dim = more expressive |
| `scaling` | `head_dim^-0.5` | Fixed 1/16, learned | Attention temperature |

**Gated Attention** (line 173, 208):
```python
xq, gate = torch.chunk(self.wq(x), 2, dim=-1)  # Q projection includes gate
output = output * torch.sigmoid(gate)           # Gated output
```

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| Gating | Enabled | Enabled, Disabled | Gating helps gradient flow |
| Gate init | Part of wq | Separate projection | Decoupled gate may help |

**QK Normalization** (line 141-142, 182-183):
```python
self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=model_args.norm_eps)
self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=model_args.norm_eps)
```

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| QK norm | Enabled | Enabled, Disabled | Stabilizes attention, may limit expressiveness |
| Norm type | ZeroCenteredRMSNorm | Standard RMSNorm, LayerNorm | Different normalization behaviors |

#### 3.1.2 Linear Attention (GatedDeltaNet)

**Location**: `model/model.py:212-577` (GatedDeltaNet class)

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `linear_num_key_heads` | 16 | 8, 16, 32 | Key head count |
| `linear_num_value_heads` | 32 | 16, 32, 64 | Value head count (can differ) |
| `linear_key_head_dim` | 128 | 64, 128, 256 | Key dimensionality |
| `linear_value_head_dim` | 128 | 64, 128, 256 | Value dimensionality |
| `linear_conv_kernel_dim` | 4 | 2, 4, 8, 16 | Local context window |

**Gated Delta Rule** (line 350-352):
```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `dt_bias` init | 1.0 | 0.5, 1.0, 2.0 | Controls decay rate |
| `A_log` init | U(0,16).log() | Different ranges | State transition strength |

---

### 3.2 Hybrid Attention Pattern

**Location**: `model/args.py:71-96`, `model/model.py:601-605`

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `full_attention_interval` | 4 | 2, 4, 6, 8, ∞ | More full attn = better quality, more compute |
| `decoder_sparse_step` | 1 | 1, 2, 4 | MoE frequency (1=all layers) |

**Layer Type Pattern**:
```python
# Default: [linear, linear, linear, full, linear, linear, linear, full, ...]
# Ratio: 75% linear attention, 25% full attention
```

| Pattern | Description | Compute | Expected Quality |
|---------|-------------|---------|------------------|
| All full | No linear attention | High | Highest |
| 1:3 ratio | 1 full : 3 linear | Medium | Good |
| 1:7 ratio | 1 full : 7 linear | Low | Acceptable |
| All linear | No full attention | Lowest | Degraded |

**Custom Layer Patterns**:
```python
# Ablation: front-loaded full attention
layer_types_frontloaded = ["full_attention"] * 12 + ["linear_attention"] * 36

# Ablation: back-loaded full attention
layer_types_backloaded = ["linear_attention"] * 36 + ["full_attention"] * 12

# Ablation: sandwich pattern
layer_types_sandwich = (
    ["full_attention"] * 6 +    # First 6 full
    ["linear_attention"] * 36 + # Middle linear
    ["full_attention"] * 6      # Last 6 full
)
```

---

### 3.3 MoE Routing

**Location**: `torchtitan/models/moe/moe.py:194-316` (TokenChoiceTopKRouter)

#### 3.3.1 Score Function

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `score_func` | "softmax" | "softmax", "sigmoid" | Softmax = competitive, Sigmoid = independent |

**Implementation** (line 265-270):
```python
if self.score_func == "sigmoid":
    scores = torch.sigmoid(scores.to(torch.float32))
elif self.score_func == "softmax":
    scores = F.softmax(scores.to(torch.float32), dim=1)
```

#### 3.3.2 Score Normalization

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `route_norm` | True | True, False | Normalizing prevents score collapse |
| `route_scale` | 1.0 | 0.5, 1.0, 2.0, 4.0 | Higher scale = sharper routing |

**Implementation** (line 292-295):
```python
if self.route_norm:
    denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
    top_scores = top_scores / denominator
top_scores = top_scores * self.route_scale
```

#### 3.3.3 Score Timing

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `score_before_experts` | False | True, False | After = Qwen3 style, Before = standard |

**Implementation** (line 488-515 in MoE.forward):
```python
if self.score_before_experts:
    # Multiply BEFORE expert forward
    routed_input = routed_input * top_scores_experts_sorted.reshape(-1, 1)
    routed_output = self.experts(routed_input, ...)
else:
    # Multiply AFTER expert forward (default)
    routed_output = self.experts(routed_input, ...)
    routed_output = routed_output * top_scores_experts_sorted.reshape(-1, 1)
```

#### 3.3.4 Top-K Selection

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `top_k` | 10 | 2, 4, 6, 8, 10, 12, 16 | More experts = more capacity, less sparsity |

**Active Compute Analysis**:
| top_k | Active % | Relative Compute | Expected Loss Delta |
|-------|----------|------------------|---------------------|
| 2 | 0.4% | 0.2x | +5-10% |
| 4 | 0.8% | 0.4x | +2-5% |
| 6 | 1.2% | 0.6x | +1-2% |
| 8 | 1.6% | 0.8x | +0.5-1% |
| 10 | 2.0% | 1.0x | Baseline |
| 16 | 3.1% | 1.6x | -0.5-1% |

---

### 3.4 Expert Architecture

**Location**: `torchtitan/models/moe/moe.py:140-192` (GroupedExperts)

#### 3.4.1 Expert Count & Granularity

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `num_experts` | 512 | 64, 128, 256, 512, 1024 | More experts = finer specialization |
| `moe_inter_dim` | 512 | Adjusted to maintain compute | Larger experts = more capacity each |

**Iso-compute configurations** (keeping active params constant):
| num_experts | moe_inter_dim | Expert Size | Total Expert Params |
|-------------|---------------|-------------|---------------------|
| 64 | 4096 | 25.2M | 77.3B |
| 128 | 2048 | 12.6M | 77.3B |
| 256 | 1024 | 6.3M | 77.3B |
| 512 | 512 | 3.1M | 77.3B |
| 1024 | 256 | 1.6M | 77.3B |

#### 3.4.2 Expert Implementation

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `use_grouped_mm` | True | True, False | grouped_mm is faster but requires alignment |

**Implementation** (line 171-184):
```python
if self.use_grouped_mm:
    return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
else:
    return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
```

#### 3.4.3 Token Group Alignment

**Location**: `torchtitan/models/moe/utils.py:15-39`

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `TOKEN_GROUP_ALIGN_SIZE_M` | 8 | 8, 16, 32 | Affects padding overhead |

---

### 3.5 Shared Experts

**Location**: `torchtitan/models/moe/moe.py:400-407, 497-509`

#### 3.5.1 Shared Expert Count

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `num_shared_experts` | 1 | 0, 1, 2, 4 | Shared experts capture common patterns |

**Implementation** (line 400-407):
```python
self.shared_experts = (
    FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
    if moe_args.num_shared_experts > 0
    else None
)
```

#### 3.5.2 Shared Expert Gating

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `shared_gate` | True | True, False | Learnable gate vs always-on |

**Implementation** (line 405-406, 503-506):
```python
self.shared_gate = nn.Linear(dim, 1, bias=False) if moe_args.shared_gate else None

# In forward:
if self.shared_gate is not None:
    shared_gate_val = F.sigmoid(self.shared_gate(flat_x))
    out = shared_out * shared_gate_val
```

#### 3.5.3 Shared Expert Dimension

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `hidden_dim` | 5120 | 2560, 5120, 10240 | Larger shared = more common capacity |

**Analysis**:
- Current shared FFN: 3 × 2048 × 5120 = 31.5M params per layer
- Current routed expert: 3 × 2048 × 512 = 3.1M params per expert
- Ratio: Shared is ~10× larger than a single routed expert

---

### 3.6 Load Balancing

**Location**: `torchtitan/models/moe/moe.py:410-429`

#### 3.6.1 Auxiliary-Loss-Free Load Balancing

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `load_balance_coeff` | 1e-3 | None, 1e-4, 1e-3, 1e-2, 1e-1 | Higher = more balanced, may hurt quality |

**Implementation** (line 410-423):
```python
# Expert bias for load balancing (auxiliary-loss-free method)
if self.load_balance_coeff is not None:
    self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))

# In router forward (line 275-279):
if expert_bias is not None:
    _, selected_experts_indices = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
```

**Note**: This implements the auxiliary-loss-free load balancing from [arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664). The `expert_bias` is updated in an optimizer pre-hook based on `tokens_per_expert`.

#### 3.6.2 Alternative: Auxiliary Loss

```python
# Potential ablation: traditional auxiliary loss
def compute_aux_loss(router_probs, expert_assignments):
    # Load balancing loss from Switch Transformer
    P_i = router_probs.mean(dim=0)  # Mean probability per expert
    f_i = expert_assignments.float().mean(dim=0)  # Fraction of tokens per expert
    return (P_i * f_i).sum() * num_experts
```

---

### 3.7 Normalization

**Location**: `model/model.py:107-124` (ZeroCenteredRMSNorm)

#### 3.7.1 Norm Type

| Norm Type | Formula | Default |
|-----------|---------|---------|
| ZeroCenteredRMSNorm | `norm(x) * (1 + w)` where `w` init to 0 | Yes |
| Standard RMSNorm | `norm(x) * w` where `w` init to 1 | Ablation |
| LayerNorm | `(x - mean) / std * w + b` | Ablation |

**Implementation** (line 116-121):
```python
def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())  # Note: (1 + weight)
    return output.type_as(x)
```

#### 3.7.2 Norm Epsilon

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `norm_eps` | 1e-6 | 1e-8, 1e-6, 1e-5 | Affects numerical stability |

---

### 3.8 Positional Encoding

**Location**: `model/model.py:44-93` (RoPE functions)

#### 3.8.1 RoPE Configuration

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `rope_theta` | 1000000 | 10000, 50000, 500000, 1000000 | Higher = better long context |
| `partial_rotary_factor` | 0.25 | 0.25, 0.5, 1.0 | How much of head_dim gets RoPE |

**Implementation** (line 84-93):
```python
rotary_dim = int(head_dim * partial_ratio)  # Only rotate this portion
xq_rot = (xq[..., :rotary_dim] * cos) + (rotate_half(xq[..., :rotary_dim]) * sin)
xq_out = torch.cat([xq_rot, xq[..., rotary_dim:]], dim=-1)  # Concat rotated + non-rotated
```

**Analysis** (default: partial_rotary_factor=0.25):
- head_dim = 256
- rotary_dim = 64 (only 64 dimensions get RoPE)
- Remaining 192 dimensions are NOT rotated

---

### 3.9 Initialization

**Location**: `model/model.py:621-624, 687-703`, `torchtitan/models/moe/moe.py:23-24, 186-191, 307-315`

#### 3.9.1 Depth-Dependent Initialization

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `depth_init` | True | True, False | Layer-dependent std prevents gradient issues |

**Implementation** (line 621-624):
```python
if model_args.depth_init:
    self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5  # Decreases with depth
else:
    self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5  # Same for all
```

#### 3.9.2 MoE Initialization

**Expert init** (line 186-191 in moe.py):
```python
def init_weights(self, init_std: float, n_layers: int):
    std_in = moe_init_std(self.w1.shape[-1], n_layers)  # (2 / (dim_in * n_layers)) ** 0.5
    nn.init.trunc_normal_(self.w1, mean=0.0, std=std_in)
    nn.init.trunc_normal_(self.w2, mean=0.0, std=std_in)
    nn.init.trunc_normal_(self.w3, mean=0.0, std=std_out)
```

**Router init** (line 307-315 in moe.py):
```python
def init_weights(self, init_std: float, n_layers: int):
    # Unit norm initialization
    nn.init.normal_(temp_weight, mean=0.0, std=1.0)
    row_norms = torch.norm(temp_weight, dim=1, keepdim=True)
    temp_weight = temp_weight / row_norms.clamp(min=1e-6)
    self.gate.weight.data = temp_weight * std
```

---

### 3.10 FFN Architecture

**Location**: `model/model.py:579-592` (FeedForward), `torchtitan/models/moe/moe.py:51-81`

#### 3.10.1 Activation Function

| Parameter | Default | Ablation Values | Hypothesis |
|-----------|---------|-----------------|------------|
| `hidden_act` | "silu" | "silu", "gelu", "relu" | SiLU (swish) is current standard |

**SwiGLU Implementation** (line 586-587):
```python
def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwiGLU
```

#### 3.10.2 FFN Expansion Ratio

| Parameter | Current Ratio | Ablation Values | Hypothesis |
|-----------|---------------|-----------------|------------|
| Shared FFN | 5120/2048 = 2.5x | 2x, 2.5x, 4x, 8/3x | Standard is 8/3 ≈ 2.67 |
| Expert FFN | 512/2048 = 0.25x | 0.125x, 0.25x, 0.5x, 1x | Very small experts! |

**Critical Observation**: The expert FFN has a **contraction** (0.25x) rather than expansion. This is unusual and worth ablating.

---

## 4. Ablation Configurations

### 4.1 High Priority Ablations

```python
# Add to qwen3next_configs in __init__.py

# =============================================================================
# HIGH PRIORITY: Load Balancing
# =============================================================================
"ablation_lb_none": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        load_balance_coeff=None,  # No load balancing
    )
),
"ablation_lb_1e4": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        load_balance_coeff=1e-4,  # Weak
    )
),
"ablation_lb_1e2": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        load_balance_coeff=1e-2,  # Strong
    )
),

# =============================================================================
# HIGH PRIORITY: Expert Granularity (iso-compute)
# =============================================================================
"ablation_experts_64_large": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=2048,  # 4x larger experts
    moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=64, num_shared_experts=1, top_k=4,
    )
),
"ablation_experts_128_medium": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=1024,  # 2x larger experts
    moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=128, num_shared_experts=1, top_k=6,
    )
),
"ablation_experts_512_small": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=256,  # 0.5x smaller experts
    moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=512, num_shared_experts=1, top_k=10,
    )
),

# =============================================================================
# HIGH PRIORITY: Top-K Selection
# =============================================================================
"ablation_topk_2": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=2)
),
"ablation_topk_4": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=4)
),
"ablation_topk_8": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_topk_16": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=16)
),
```

### 4.2 Medium Priority Ablations

```python
# =============================================================================
# MEDIUM PRIORITY: Routing Mechanism
# =============================================================================
"ablation_routing_sigmoid": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        score_func="sigmoid",  # vs default softmax
        route_norm=False,
    )
),
"ablation_routing_no_norm": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        score_func="softmax",
        route_norm=False,  # No normalization
    )
),
"ablation_routing_scale_2": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        route_scale=2.0,  # Sharper routing
    )
),
"ablation_score_before": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1, top_k=8,
        score_before_experts=True,  # vs default False
    )
),

# =============================================================================
# MEDIUM PRIORITY: Shared Experts
# =============================================================================
"ablation_shared_0": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=0,  # No shared experts
        top_k=8,
    )
),
"ablation_shared_2": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=1280,  # Smaller per shared expert
    moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=2,  # Two shared experts
        top_k=8,
    )
),
"ablation_shared_gate_off": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(
        num_experts=256, num_shared_experts=1,
        shared_gate=False,  # Always-on shared expert
        top_k=8,
    )
),

# =============================================================================
# MEDIUM PRIORITY: Hybrid Attention Pattern
# =============================================================================
"ablation_all_full_attn": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    full_attention_interval=1,  # All layers use full attention
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_attn_ratio_1_7": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    full_attention_interval=8,  # 1:7 full:linear ratio
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_attn_ratio_1_1": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    full_attention_interval=2,  # 1:1 full:linear ratio
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
```

### 4.3 Lower Priority Ablations

```python
# =============================================================================
# LOWER PRIORITY: Positional Encoding
# =============================================================================
"ablation_rope_theta_10k": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    rope_theta=10000.0,  # vs default 1000000
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_full_rotary": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    partial_rotary_factor=1.0,  # vs default 0.25
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_half_rotary": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    partial_rotary_factor=0.5,  # vs default 0.25
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),

# =============================================================================
# LOWER PRIORITY: Attention Configuration
# =============================================================================
"ablation_gqa_4": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=4,  # GQA 2:1 vs default 8:1
    head_dim=128, hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_mha": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=8,  # MHA (no GQA)
    head_dim=128, hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),

# =============================================================================
# LOWER PRIORITY: Initialization
# =============================================================================
"ablation_no_depth_init": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    depth_init=False,  # vs default True
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),

# =============================================================================
# LOWER PRIORITY: Linear Attention Parameters
# =============================================================================
"ablation_linear_conv_8": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    linear_conv_kernel_dim=8,  # vs default 4
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
"ablation_linear_heads_64": Qwen3NextModelArgs(
    dim=1024, n_layers=24, n_heads=8, n_kv_heads=2, head_dim=128,
    hidden_dim=2560, moe_inter_dim=512, moe_enabled=True,
    linear_num_value_heads=64,  # vs default 32
    moe_args=MoEArgs(num_experts=256, num_shared_experts=1, top_k=8)
),
```

---

## 5. Metrics & Analysis

### 5.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Loss** | Training/validation loss | Lower is better |
| **Perplexity** | exp(loss) | Lower is better |
| **Throughput** | Tokens/second | Higher is better |
| **MFU** | Model FLOP utilization | Higher is better |

### 5.2 MoE-Specific Metrics

```python
moe_metrics = {
    # Expert utilization
    "expert_utilization_mean": "tokens_per_expert.mean()",
    "expert_utilization_std": "tokens_per_expert.std()",
    "expert_utilization_cv": "std / mean (coefficient of variation)",
    "expert_utilization_max": "tokens_per_expert.max()",
    "expert_utilization_min": "tokens_per_expert.min()",
    "expert_utilization_gini": "Gini coefficient of utilization",

    # Routing entropy
    "routing_entropy": "-sum(p * log(p)) for routing probabilities",
    "routing_entropy_normalized": "entropy / log(num_experts)",

    # Expert collapse detection
    "dead_experts_count": "experts with < 1% of mean utilization",
    "dominant_experts_count": "experts with > 10x mean utilization",

    # Load balance
    "load_balance_loss": "aux_loss if using traditional LB",
    "expert_bias_norm": "||expert_bias||_2 for aux-loss-free LB",
}
```

### 5.3 Attention Metrics (for hybrid attention ablations)

```python
attention_metrics = {
    # Full attention
    "attention_entropy": "Entropy of attention weights",
    "attention_sparsity": "Fraction of near-zero attention weights",

    # Linear attention
    "delta_rule_gate_mean": "Mean of beta (gating value)",
    "delta_rule_decay_mean": "Mean of g (decay rate)",
}
```

### 5.4 Gradient Metrics

```python
gradient_metrics = {
    "grad_norm_total": "Total gradient norm",
    "grad_norm_attention": "Attention gradient norm",
    "grad_norm_ffn": "FFN gradient norm",
    "grad_norm_router": "Router gradient norm",
    "grad_norm_experts": "Expert gradient norm",
    "grad_norm_shared": "Shared expert gradient norm",
}
```

---

## 6. Priority & Timeline

### 6.1 Priority Matrix

| Priority | Category | Ablations | Expected Impact |
|----------|----------|-----------|-----------------|
| **P0** | Load Balance | lb_none, lb_1e4, lb_1e2 | Critical for training stability |
| **P0** | Top-K | topk_2, topk_4, topk_8, topk_16 | Direct quality/compute trade-off |
| **P0** | Expert Granularity | experts_64, experts_128, experts_512 | Fundamental architecture choice |
| **P1** | Routing | sigmoid, no_norm, scale_2, score_before | Routing efficiency |
| **P1** | Shared Experts | shared_0, shared_2, shared_gate_off | Common pattern learning |
| **P1** | Hybrid Attention | all_full, 1:7, 1:1 | Compute/quality trade-off |
| **P2** | Positional | rope_theta, partial_rotary | Long context performance |
| **P2** | Attention Config | gqa_4, mha | Memory/quality trade-off |
| **P3** | Initialization | no_depth_init | Training stability |
| **P3** | Linear Attention | conv_8, heads_64 | Linear attention efficiency |

### 6.2 Recommended Ablation Order

```
PHASE 1 (Days 1-3): Critical Architecture
├── Load balance coefficient sweep (3 runs)
├── Top-K sweep (4 runs)
└── Expert granularity sweep (3 runs)

PHASE 2 (Days 4-6): Routing & Experts
├── Routing mechanism ablations (4 runs)
└── Shared expert ablations (3 runs)

PHASE 3 (Days 7-8): Attention Pattern
└── Hybrid attention ratio ablations (3 runs)

PHASE 4 (Days 9-10): Secondary Parameters
├── Positional encoding (2 runs)
└── Attention configuration (2 runs)

PHASE 5 (Day 11+): Scale Validation
└── Top 2-3 configs trained to 100B+ tokens
```

### 6.3 Compute Budget Allocation

| Phase | % of Budget | Purpose |
|-------|-------------|---------|
| Phase 1-4 | 15-20% | Small-scale ablations (~10B tokens each) |
| Phase 5 | 20-25% | Scale validation (~100B tokens each) |
| Final Training | 55-65% | Full training run |

---

## 7. Training Configuration for Ablations

### 7.1 Critical Batch Size Analysis

Based on MoE scaling laws (particularly from the Unified Scaling Laws for Routed Models), the critical batch size determines the minimum effective batch size needed for stable, efficient training.

#### Theoretical Framework

The critical batch size follows:
```
B_crit = B_* / L

Where:
- B_* ≈ 1e8 to 2e8 tokens (empirical constant)
- L = current training loss
```

#### Practical Batch Size Recommendations

For ablation models (~8B total, ~0.8B active params, 256 experts, top-8):

| Loss Range | Theoretical B_crit | Practical Minimum | Recommended |
|------------|-------------------|-------------------|-------------|
| L ≈ 3.0 (early) | ~50M tokens | 0.5M tokens | 0.5-1M tokens |
| L ≈ 2.5 (mid) | ~60M tokens | 0.5M tokens | 0.5-1M tokens |
| L ≈ 2.0 (late) | ~75M tokens | 1M tokens | 1-2M tokens |

**Key insight**: Theoretical B_crit is very large (~50-90M tokens), but practical training works well at ~1-2% of this due to:
1. Gradient noise helps exploration
2. Compute efficiency favors smaller batches
3. Numerical stability constraints

### 7.2 Recommended Training Configuration

#### Single Node (8 GPUs) Setup

```toml
# Recommended ablation config for stable training

[training]
local_batch_size = 4
seq_len = 4096
gradient_accumulation_steps = 8

# Effective batch size calculation:
# 8 GPUs × 4 local × 8 accum × 4096 seq = 1,048,576 tokens/step ✓
```

#### Configuration Matrix by Model Scale

| Model Scale | Active Params | local_batch | grad_accum | Effective Batch |
|-------------|---------------|-------------|------------|-----------------|
| Ablation Small | ~0.8B | 4 | 4 | ~0.5M tokens |
| Ablation Medium | ~1.5B | 4 | 8 | ~1M tokens |
| Ablation Large | ~3B | 4 | 8-16 | ~1-2M tokens |
| Full 80B_A3B | ~4.6B | 4 | 16 | ~2M tokens |

### 7.3 Learning Rate Scaling

When changing batch size, scale learning rate appropriately:

```
LR_new = LR_base × sqrt(batch_new / batch_base)
```

| Effective Batch | Recommended Base LR |
|-----------------|---------------------|
| 0.5M tokens | 2e-4 |
| 1M tokens | 3e-4 |
| 2M tokens | 4e-4 |

### 7.4 Example Training Config

```toml
# /home/phuc/workspace/moe/moe_ablations/configs/ablation_baseline.toml

[job]
description = "Ablation baseline training"

[training]
local_batch_size = 4
seq_len = 4096
gradient_accumulation_steps = 8
train_steps = 2500  # ~2.5B tokens
warmup_steps = 250
checkpoint_interval = 500

[optimizer]
name = "adamw"
lr = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

[lr_scheduler]
name = "cosine"
warmup_steps = 250
min_lr_ratio = 0.1

[model]
name = "qwen3_next"
flavor = "ablation_routing_baseline"  # or other ablation config
```

### 7.5 Token Budget Guidelines

| Phase | Tokens per Run | Purpose |
|-------|----------------|---------|
| Quick validation | 1-2B | Verify training stability |
| Ablation sweep | 5-10B | Compare configurations |
| Scale validation | 50-100B | Validate scaling behavior |
| Full training | 500B+ | Production model |

---

## Appendix A: Key Code References

### A.1 MoE Forward Pass Flow

```
Input: x (bs, slen, dim)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. Router (TokenChoiceTopKRouter)               │
│    scores = gate(x)                             │
│    scores = softmax/sigmoid(scores)             │
│    top_scores, indices = topk(scores + bias, k) │
│    if route_norm: top_scores /= sum(top_scores) │
│    top_scores *= route_scale                    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 2. Token Reorderer                              │
│    Sort tokens by expert assignment             │
│    num_tokens_per_expert = histc(indices)       │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 3. Expert Computation (GroupedExperts)          │
│    if score_before_experts:                     │
│        routed_input *= scores                   │
│    routed_output = experts(routed_input)        │
│    if not score_before_experts:                 │
│        routed_output *= scores                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 4. Shared Expert (always runs in parallel)      │
│    shared_out = shared_experts(x)               │
│    if shared_gate:                              │
│        shared_out *= sigmoid(shared_gate(x))    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 5. Combine                                      │
│    out = shared_out + scatter_add(routed_output)│
└─────────────────────────────────────────────────┘
    │
    ▼
Output: out (bs, slen, dim)
```

### A.2 Attention Forward Pass Flow (Full Attention)

```
Input: x (bs, seqlen, dim)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. Projections                                  │
│    xq, gate = chunk(wq(x), 2)  # Q includes gate│
│    xk = wk(x)                                   │
│    xv = wv(x)                                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 2. QK Normalization                             │
│    xq = q_norm(xq)                              │
│    xk = k_norm(xk)                              │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 3. RoPE (partial)                               │
│    xq, xk = apply_rotary_emb(xq, xk,           │
│                 partial_ratio=0.25)             │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 4. GQA Expansion                                │
│    xk = repeat_kv(xk, n_rep=8)                  │
│    xv = repeat_kv(xv, n_rep=8)                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 5. Attention (FlexAttention)                    │
│    output = flex_attn(xq, xk, xv, mask)         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 6. Gated Output                                 │
│    output = output * sigmoid(gate)              │
│    output = wo(output)                          │
└─────────────────────────────────────────────────┘
    │
    ▼
Output: output (bs, seqlen, dim)
```

---

## Appendix B: Expected Outcomes Summary

| Ablation | Expected Finding |
|----------|------------------|
| **Load Balance** | 1e-3 optimal; None causes collapse; 1e-2+ hurts quality |
| **Top-K** | Diminishing returns above k=8; k=2-4 too sparse for quality |
| **Expert Granularity** | 128-256 experts likely optimal at this scale |
| **Routing: sigmoid** | May help with dead experts but harder to balance |
| **Routing: no_norm** | Likely unstable, high variance |
| **Routing: scale=2** | Sharper routing, may cause expert collapse |
| **Score Before** | Standard MoE behavior, compare with Qwen3 (after) |
| **Shared: 0** | Significant quality drop, proves shared experts valuable |
| **Shared: 2** | Similar to 1 unless very different dim |
| **Shared Gate Off** | Slightly worse, gate helps adaptive scaling |
| **Hybrid: All Full** | Best quality, highest compute |
| **Hybrid: 1:7** | Quality drop, faster training |
| **RoPE theta** | Higher theta better for long context |
| **Partial Rotary** | 0.25 may be suboptimal, test 0.5 |
| **GQA Ratio** | 8:1 aggressive, 4:1 may be better |
| **Depth Init** | Important for deep models, keep enabled |

---

*Document Version: 1.0*
*Last Updated: Based on code analysis of Qwen3-Next implementation*
