# Comprehensive Ablation Plan: 80B_A3B Qwen3-Next MoE

## Architecture Summary

| Parameter | Value |
|-----------|-------|
| Total Params | 80.4B |
| Active Params | 4.6B |
| Sparsity | 17.5x |
| dim | 2048 |
| n_layers | 48 |
| n_heads / n_kv_heads | 16 / 2 |
| head_dim | 256 |
| hidden_dim (shared FFN) | 5120 |
| moe_inter_dim (expert) | 512 |
| num_experts | 512 |
| top_k | 10 |
| num_shared_experts | 1 |

**Memory**: 153GB/163GB per GPU (8× B200 single node) - tight fit!

---

## Ablation Strategy Overview

### Budget Allocation (assuming 100% = full training compute)

| Phase | Compute | Purpose |
|-------|---------|---------|
| Phase 1 | 2-3% | Small-scale validation & hyperparameter sweep |
| Phase 2 | 5-8% | Architecture decisions (routing, experts) |
| Phase 3 | 10-15% | Scale validation & final selection |
| Phase 4 | 75-80% | Full training run |

---

## Phase 1: Small-Scale Validation (2-3% compute)

### 1.1 Baseline Runs at 1/10 Scale
Train smaller proxies to validate scaling behavior.

```python
# Proxy config: ~8B total, ~0.5B active
"8B_A0.5B_proxy": Qwen3NextModelArgs(
    dim=1024,
    n_layers=24,
    n_heads=8,
    n_kv_heads=2,
    head_dim=128,
    hidden_dim=2560,
    moe_inter_dim=256,
    moe_args=MoEArgs(
        num_experts=256,
        num_shared_experts=1,
        top_k=5,
    )
)
```

**Runs** (each ~10B tokens):
- [ ] Dense baseline (no MoE)
- [ ] MoE baseline with default config
- [ ] Verify loss curve follows expected scaling

### 1.2 Learning Rate Sweep
Critical for MoE stability.

| Run | Base LR | Notes |
|-----|---------|-------|
| 1 | 1e-4 | Conservative |
| 2 | 3e-4 | Default |
| 3 | 5e-4 | Aggressive |
| 4 | 1e-3 | Very aggressive |

**Metric**: Loss at 5B tokens, gradient norm stability

---

## Phase 2: Architecture Decisions (5-8% compute)

### 2.1 Routing Mechanism Ablations

#### 2.1.1 Score Function
```python
# Ablation configs
score_func_ablations = [
    {"score_func": "softmax", "route_norm": True},   # Current (Qwen3 style)
    {"score_func": "sigmoid", "route_norm": False},  # DeepSeek style
    {"score_func": "softmax", "route_norm": False},  # Standard MoE
]
```

| Config | Expected Behavior |
|--------|-------------------|
| softmax + norm | Normalized competition, stable routing |
| sigmoid | Independent expert activation, can have >1 "winner" |
| softmax no-norm | Sharp competition, potential collapse |

#### 2.1.2 Route Scale
```python
route_scale_ablations = [0.5, 1.0, 2.0, 4.0]
```
Higher scale → sharper routing → potential expert collapse

#### 2.1.3 Score Timing
```python
score_before_experts_ablations = [True, False]
```
- `True`: Compute scores before expert forward (standard)
- `False`: Compute scores after (Qwen3 default) - may help with expert specialization

### 2.2 Expert Count & Granularity

**Key Trade-off**: More experts = finer granularity but higher memory, more routing overhead

| Config | Experts | top_k | Sparsity | Expert Size | Memory Impact |
|--------|---------|-------|----------|-------------|---------------|
| Fine-grained | 512 | 10 | 17.5x | 3.1M | Baseline |
| Medium | 256 | 8 | 11x | 6.3M | -15% experts |
| Coarse | 128 | 6 | 7x | 12.6M | -30% experts |
| Very coarse | 64 | 4 | 5x | 25.2M | -45% experts |

```python
expert_count_ablations = [
    {"num_experts": 512, "top_k": 10, "moe_inter_dim": 512},   # Baseline
    {"num_experts": 256, "top_k": 8, "moe_inter_dim": 1024},   # Fewer, larger
    {"num_experts": 128, "top_k": 6, "moe_inter_dim": 2048},   # Even fewer
    {"num_experts": 64, "top_k": 4, "moe_inter_dim": 4096},    # Coarse
]
```

**Hypothesis**: Finer granularity helps specialization but increases routing noise

### 2.3 Top-K Selection

| top_k | Active Expert % | Compute | Expected Effect |
|-------|-----------------|---------|-----------------|
| 4 | 0.8% | Lower | More sparse, potential quality loss |
| 8 | 1.6% | Medium | Balanced |
| 10 | 2.0% | Baseline | Current config |
| 16 | 3.1% | Higher | More capacity, less sparse |

```python
top_k_ablations = [4, 6, 8, 10, 12, 16]
```

**Key metric**: Loss vs compute trade-off curve

### 2.4 Shared Expert Configuration

#### 2.4.1 Number of Shared Experts
```python
shared_expert_ablations = [
    {"num_shared_experts": 0, "hidden_dim": 5120},   # No shared
    {"num_shared_experts": 1, "hidden_dim": 5120},   # Baseline
    {"num_shared_experts": 2, "hidden_dim": 2560},   # More shared, smaller each
    {"num_shared_experts": 4, "hidden_dim": 1280},   # Many small shared
]
```

#### 2.4.2 Shared Gate
```python
shared_gate_ablations = [True, False]
```
- `True`: Shared expert has learnable gate weight
- `False`: Shared expert always fully activated

### 2.5 Load Balancing

#### 2.5.1 Auxiliary Loss Coefficient
```python
load_balance_ablations = [
    {"load_balance_coeff": None},      # No aux loss
    {"load_balance_coeff": 1e-4},      # Very weak
    {"load_balance_coeff": 1e-3},      # Baseline
    {"load_balance_coeff": 1e-2},      # Strong
    {"load_balance_coeff": 1e-1},      # Very strong
]
```

**Metrics to track**:
- Expert utilization variance
- Routing entropy
- Token drop rate (if using capacity factor)

---

## Phase 3: Scale Validation (10-15% compute)

### 3.1 Selected Configurations
Based on Phase 2 results, train 2-3 promising configs to ~100B tokens:

```python
# Example candidates
candidates = [
    "baseline_80B_A3B",           # Original config
    "optimized_routing",          # Best routing from Phase 2
    "optimized_expert_count",     # Best expert config from Phase 2
]
```

### 3.2 Scaling Law Validation
Fit loss curves to:
```
L(D) = E + A/N^α + B/D^β
```

Validate predictions match empirical results.

### 3.3 Training Dynamics Analysis

**Monitor throughout training**:
1. **Expert utilization** per layer (should be balanced)
2. **Routing entropy** (should remain high, not collapse)
3. **Gradient norms** per component (attention vs experts vs shared)
4. **Loss per expert** (if possible)

---

## Phase 4: Full Training

### 4.1 Final Configuration
Based on Phase 3 results.

### 4.2 Training Schedule
```
Stage 1 (60%): Web corpus, standard LR schedule
Stage 2 (25%): Code + high-quality data, lower LR
Stage 3 (10%): Instruction data, very low LR
Stage 4 (5%):  Final annealing
```

### 4.3 Checkpointing Strategy
- Save every 10B tokens
- Full checkpoint + expert statistics
- Track per-layer expert utilization

---

## Ablation Configurations Code

Add these to `__init__.py`:

```python
# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

# Baseline for ablations (smaller scale for faster iteration)
ablation_base = {
    "dim": 1024,
    "n_layers": 24,
    "n_heads": 8,
    "n_kv_heads": 2,
    "head_dim": 128,
    "hidden_dim": 2560,
    "moe_inter_dim": 512,
}

qwen3next_ablation_configs = {
    # =================================================================
    # ROUTING ABLATIONS
    # =================================================================
    "ablation_score_softmax_norm": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
        )
    ),
    "ablation_score_sigmoid": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            top_k=8,
            score_func="sigmoid",
            route_norm=False,
            route_scale=1.0,
        )
    ),
    "ablation_route_scale_2": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=2.0,
        )
    ),

    # =================================================================
    # EXPERT COUNT ABLATIONS (iso-compute: adjust moe_inter_dim)
    # =================================================================
    "ablation_experts_512_topk10": Qwen3NextModelArgs(
        **ablation_base,
        moe_inter_dim=256,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=512,
            top_k=10,
        )
    ),
    "ablation_experts_256_topk8": Qwen3NextModelArgs(
        **ablation_base,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            top_k=8,
        )
    ),
    "ablation_experts_128_topk6": Qwen3NextModelArgs(
        **ablation_base,
        moe_inter_dim=1024,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=128,
            top_k=6,
        )
    ),
    "ablation_experts_64_topk4": Qwen3NextModelArgs(
        **ablation_base,
        moe_inter_dim=2048,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=64,
            top_k=4,
        )
    ),

    # =================================================================
    # TOP-K ABLATIONS (fixed expert count)
    # =================================================================
    "ablation_topk_4": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=4)
    ),
    "ablation_topk_8": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8)
    ),
    "ablation_topk_12": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=12)
    ),
    "ablation_topk_16": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=16)
    ),

    # =================================================================
    # LOAD BALANCE ABLATIONS
    # =================================================================
    "ablation_lb_none": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, load_balance_coeff=None)
    ),
    "ablation_lb_1e4": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, load_balance_coeff=1e-4)
    ),
    "ablation_lb_1e3": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, load_balance_coeff=1e-3)
    ),
    "ablation_lb_1e2": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, load_balance_coeff=1e-2)
    ),

    # =================================================================
    # SHARED EXPERT ABLATIONS
    # =================================================================
    "ablation_shared_0": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, num_shared_experts=0)
    ),
    "ablation_shared_1": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, num_shared_experts=1)
    ),
    "ablation_shared_2": Qwen3NextModelArgs(
        **ablation_base,
        hidden_dim=1280,  # Smaller shared FFN
        moe_enabled=True,
        moe_args=MoEArgs(num_experts=256, top_k=8, num_shared_experts=2)
    ),

    # =================================================================
    # DENSE BASELINE
    # =================================================================
    "ablation_dense_baseline": Qwen3NextModelArgs(
        **ablation_base,
        moe_enabled=False,
    ),
}
```

---

## Metrics & Logging

### Required Metrics per Run

```python
metrics_to_log = {
    # Loss metrics
    "loss": "Training loss",
    "val_loss": "Validation loss",

    # Expert metrics (per layer)
    "expert_utilization_mean": "Mean tokens per expert",
    "expert_utilization_std": "Std of tokens per expert",
    "expert_utilization_max": "Max tokens to single expert",
    "expert_utilization_min": "Min tokens to single expert",
    "routing_entropy": "Entropy of routing distribution",

    # Gradient metrics
    "grad_norm_total": "Total gradient norm",
    "grad_norm_router": "Router gradient norm",
    "grad_norm_experts": "Expert gradient norm",
    "grad_norm_shared": "Shared expert gradient norm",

    # Throughput
    "tokens_per_second": "Training throughput",
    "mfu": "Model FLOP utilization",
}
```

### Analysis Scripts

Create analysis notebook at:
`/home/phuc/workspace/moe/moe_ablations/analysis/ablation_analysis.ipynb`

---

## Priority Ranking

### High Priority (Do First)
1. **Load balance coefficient** - Critical for preventing expert collapse
2. **top_k selection** - Directly affects compute/quality trade-off
3. **Expert granularity** - Fundamental architecture decision

### Medium Priority
4. **Score function** (softmax vs sigmoid)
5. **Shared expert count**
6. **Route scale**

### Lower Priority (If Compute Allows)
7. **Score timing** (before/after experts)
8. **Shared gate** configuration
9. **Router architecture** variations

---

## Expected Outcomes

| Ablation | Expected Finding |
|----------|------------------|
| Load balance | 1e-3 optimal, lower causes collapse, higher hurts loss |
| top_k | Diminishing returns above k=8, k=4 too sparse |
| Experts | 256-512 optimal for this scale, 64 too coarse |
| Score func | softmax+norm most stable |
| Shared experts | 1-2 shared helps, 0 hurts generalization |

---

## Timeline Estimate

| Phase | Duration | Runs |
|-------|----------|------|
| Phase 1 | 2-3 days | 6-8 runs |
| Phase 2 | 5-7 days | 15-20 runs |
| Phase 3 | 3-4 days | 2-3 runs |
| Phase 4 | 2-3 weeks | 1 run |

**Total**: ~4 weeks for comprehensive ablation study

---

## References

- [Unified Scaling Laws for Routed Models](https://arxiv.org/abs/2202.01169) - Clark et al.
- [DeepSeek-MoE](https://arxiv.org/abs/2401.06066) - Expert granularity insights
- [Mixtral](https://arxiv.org/abs/2401.04088) - Practical MoE training
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Load balancing strategies
