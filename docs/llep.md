# Least-Loaded Expert Parallelism (LLEP)

LLEP is a dynamic load-balancing strategy for Expert Parallelism (EP) in Mixture-of-Experts (MoE) models. It redistributes expert workloads across GPUs at runtime to handle the token imbalance inherent in models like DeepSeek-V3, Kimi-K2, and Qwen3.

**Reference:** "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts" (Nguyen et al., Salesforce AI Research)

## How It Works

Standard EP assigns each expert to a fixed GPU. When routing is imbalanced (e.g., 80% of tokens go to 2 of 256 experts), the "hot" GPUs become bottlenecks while others sit idle.

LLEP fixes this by:

1. **Gathering** per-rank expert token counts via `all_gather`
2. **Planning** a Longest-Processing-Time (LPT) assignment that splits overloaded experts across multiple GPUs
3. **Transferring** expert weights to "helper" GPUs via P2P
4. **Dispatching** tokens via a modified AllToAll that routes to the helper GPUs
5. **Computing** SwiGLU FFN on each GPU's assigned token-expert pairs
6. **Combining** results via inverse AllToAll back to the original token order

## Architecture

### Hook-Based Flow (Recommended)

The hook-based API decomposes LLEP into three steps that integrate with the existing MoE forward pass:

```python
from torchtitan.distributed.llep import (
    llep_dispatch_tokens,
    llep_compute_with_weights,
    llep_combine_output,
)

# EP pre-hook: plan + dispatch tokens
dispatched_tokens, padded_counts, state = llep_dispatch_tokens(
    routed_input, num_tokens_per_expert, ep_group,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1024,
    adaptive_threshold=1.3,
)

# Inside GroupedExperts.forward: P2P weight transfer + compute
output = llep_compute_with_weights(
    dispatched_tokens, padded_counts,
    w1, w2, w3, state,
    use_grouped_mm=True,
)

# EP post-hook: combine results
combined = llep_combine_output(output, state)
```

## Configuration

LLEP is configured via `LLEPConfig` (defined in `torchtitan/models/moe/moe.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `max_tokens_factor` | alpha | 1.1 | GPU capacity factor. `max_tokens_per_gpu = alpha * (total_tokens / num_gpus)`. Controls how much overload is tolerated before spilling. |
| `min_tokens_per_gemm` | m | 1024 | Minimum tokens to justify spilling to a helper GPU. Below this the GEMM is too small to be efficient. |
| `adaptive_threshold` | lambda | 0.0 | Imbalance ratio (`max_gpu_load / mean_gpu_load`) to trigger LLEP. Set to 0 to always use LLEP. Paper recommends 1.3. |
| `verbose` | - | false | Enable per-step distribution logging (see [Distribution Logging](#distribution-logging) below). |

### TOML Configuration

```toml
[llep]
enabled = true
max_tokens_factor = 1.1
min_tokens_per_gemm = 1024
adaptive_threshold = 1.3
verbose = false            # set true for per-step distribution logging
```

### Environment Variable Overrides

These override TOML/code values at runtime (useful for tuning without config changes):

| Variable | Overrides |
|----------|-----------|
| `EP_MAX_TOKENS_FACTOR` | `max_tokens_factor` |
| `EP_MIN_TOKENS_PER_GEMM` | `min_tokens_per_gemm` |
| `EP_ADAPTIVE_THRESHOLD` | `adaptive_threshold` |
| `LLEP_W_TRANSFER_AUTOGRAD` | Enable autograd for weight transfer (default: 1) |
| `LLEP_MERGE_A2A` | Merge hidden+scores+ids into single AllToAll (default: 1) |
| `LLEP_DEBUG` | Verbose per-step logging (default: 0). Equivalent to `[llep] verbose = true` |

## Distribution Logging

LLEP can log per-step token distribution details showing how load balancing works. Enable via TOML (`verbose = true`) or env var (`LLEP_DEBUG=1`). Logs are emitted on **all ranks** so you can see each GPU's perspective.

### Enabling

```toml
[llep]
enabled = true
verbose = true
```

Or at runtime without config changes:

```bash
LLEP_DEBUG=1 torchrun --nproc_per_node=8 -m torchtitan.train --job.config_file ...
```

### Log Messages

Each LLEP dispatch produces 4 log messages per rank:

**1. BEFORE** — Native distribution (before LLEP redistribution):
```
[LLEP rank=0 step=1] BEFORE: total_tokens=32768 imbalance=1.68
  native_gpu_loads=[44579, 25139, 54981, 22581, 29444, 29702, 25659, 30059]
  expert_counts=[3202, 8887, 5002, ...]
```

**2. AFTER LPT** — After LLEP redistribution with before->after imbalance ratio:
```
[LLEP rank=0 step=1] AFTER LPT: use_lpt=True imbalance=1.68->1.10
  llep_gpu_loads=[36044, 30735, 36044, 35922, 29444, 29702, 34194, 30059]
  weight_transfers (3):
    expert 23: GPU 2->3 (tokens 0-13341)
    expert 18: GPU 2->1 (tokens 6278-11874)
    expert 1: GPU 0->6 (tokens 352-8887)
```

**3. SEND_MATRIX** — AllToAll routing matrix (row=source GPU, col=destination GPU):
```
[LLEP rank=0 step=1] SEND_MATRIX (row=src, col=dst):
    [4697, 3206, 5152, 4494, 3665, 3670, 4023, 3861]
    [4347, 3080, 5377, 4520, 3666, 3725, 4443, 3610]
    ...
  input_splits=[4697, 3206, 5152, 4494, 3665, 3670, 4023, 3861]
  output_splits=[4697, 4347, 4430, 4400, 4382, 4483, 4627, 4678]
```

**4. RECEIVED** — What each rank actually received after AllToAll:
```
[LLEP rank=0 step=1] RECEIVED: total_recv=36044 experts=[0, 1, 2, 3, 4, 5, 6, 7] counts=[3202, 352, 5002, 7563, 3882, 8722, 1999, 5322]
```

### Debug Config for 8 GPUs

A ready-made config for inspecting LLEP distribution on 8 GPUs:

```bash
torchrun --nproc_per_node=8 -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  2>&1 | tee /tmp/llep_distribution_logs.txt
```

This config uses `debugmodel_ep8_llep` (64 experts, top_k=8, EP=8) with `min_tokens_per_gemm=1` and `adaptive_threshold=0.0` so LLEP always triggers, even at small debug scale. See `train_configs/debug_model_ep8_llep.toml`.

## Benchmark: LLEP vs Standard EP

### Model

The `debugmodel_ep8_llep` flavor is a 9.5B-parameter MoE model designed for single-node 8-GPU benchmarking:

| Parameter | Value |
|-----------|-------|
| Total params | 9.5B (8.9 GB bf16) |
| MoE expert params | 9.1B (96%) |
| Active params/token | 1.6B (top_k=8 of 64 experts) |
| dim | 2048 |
| inter_dim | 8192 |
| moe_inter_dim | 1536 |
| n_layers | 16 (1 dense + 15 MoE) |
| num_experts | 64 |
| top_k | 8 |
| EP | 8 (8 local experts/GPU) |

Training config: `lbs=6, seq_len=4096, AdamW, no compile, no activation checkpointing`.

### Reproducing

```bash
cd torchtitan

# WITH LLEP (20 steps)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --compile.no-enable \
  2>&1 | tee llep_with_llep.txt

# WITHOUT LLEP (20 steps, same model)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --compile.no-enable --llep.enabled=False \
  2>&1 | tee llep_no_llep.txt
```

To enable verbose per-step distribution logging (shows BEFORE/AFTER imbalance, send matrix, weight transfers):

```bash
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 3 --compile.no-enable --llep.verbose=True \
  2>&1 | tee llep_verbose_logs.txt
```

### Results (8xB200, 20 steps)

**Speed** (steps 5-20 average, excluding warmup):

| | With LLEP | Without LLEP | Delta |
|---|---|---|---|
| Mean TPS | ~16,200 | ~14,900 | **+8.7%** |
| Mean MFU | 8.2% | 7.5% | +9.3% |

**Memory** (per-GPU at step 20):

| | With LLEP | Without LLEP |
|---|---|---|
| Active range | 103-107 GiB (58-60%) | 93-111 GiB (52-62%) |
| Reserved range | 115-118 GiB (64-66%) | 106-168 GiB (60-**94%**) |
| Max reserved | 118 GiB | **168 GiB** (near OOM) |
| Spread (reserved) | ~3 GiB | **62 GiB** |

Without LLEP, the most-loaded GPU hits 94% reserved memory (near OOM) while the least-loaded sits at 60%. LLEP keeps all GPUs in a tight 64-66% band. LLEP is both faster (less straggler waiting from load imbalance) and safer (no GPU near OOM).

### Per-GPU Memory Breakdown (step 5)

Detailed per-GPU view showing the memory imbalance that LLEP eliminates:

**With LLEP** — all GPUs balanced within a 3 GiB band:

| GPU | Active (GiB) | Active % | Reserved (GiB) | Reserved % | TPS |
|-----|-------------|----------|----------------|------------|-----|
| 0 | 106.36 | 59.6% | 119.27 | 66.9% | 16,158 |
| 1 | 104.56 | 58.6% | 115.71 | 64.9% | 16,195 |
| 2 | 107.17 | 60.1% | 117.41 | 65.8% | 16,170 |
| 3 | 104.99 | 58.9% | 113.94 | 63.9% | 16,127 |
| 4 | 104.06 | 58.3% | 118.19 | 66.3% | 16,164 |
| 5 | 106.05 | 59.5% | 117.53 | 65.9% | 15,698 |
| 6 | 106.60 | 59.8% | 119.01 | 66.7% | 16,205 |
| 7 | 105.99 | 59.4% | 117.96 | 66.1% | 16,225 |
| **Spread** | **3.1** | | **5.3** | | |

**Without LLEP** — wildly imbalanced, one GPU near OOM:

| GPU | Active (GiB) | Active % | Reserved (GiB) | Reserved % | TPS |
|-----|-------------|----------|----------------|------------|-----|
| 0 | 120.88 | 67.8% | 133.50 | 74.9% | 13,967 |
| 1 | 123.34 | **69.2%** | 131.20 | 73.6% | 13,988 |
| 2 | 98.38 | 55.2% | 132.71 | 74.4% | 13,961 |
| 3 | 100.26 | 56.2% | 143.61 | 80.5% | 13,959 |
| 4 | 72.77 | **40.8%** | 118.05 | 66.2% | 13,987 |
| 5 | 112.71 | 63.2% | 158.73 | 89.0% | 13,962 |
| 6 | 105.93 | 59.4% | 165.68 | **92.9%** | 13,987 |
| 7 | 107.93 | 60.5% | 161.10 | **90.3%** | 13,902 |
| **Spread** | **50.6** | | **47.6** | | |

Key observations:
- Without LLEP, GPU 6 reserves **165.7 GiB (92.9%)** of 178 GiB — one more imbalanced step away from OOM.
- GPU 4 is nearly idle at 40.8% active while GPU 1 is at 69.2% — a **28.4 percentage point** gap.
- LLEP compresses the active memory spread from **50.6 GiB to 3.1 GiB** (16x reduction).
- Every GPU with LLEP runs at ~16,100+ TPS vs ~13,960 without — the straggler GPU drags everyone down.

To reproduce this comparison:

```bash
cd torchtitan

# 5-step memory comparison with LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 --compile.no-enable \
  2>&1 | tee llep_memory_with_llep.txt

# 5-step memory comparison without LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 --compile.no-enable --llep.enabled=False \
  2>&1 | tee llep_memory_no_llep.txt

# Extract per-GPU memory at step 5
grep "step:  5" llep_memory_with_llep.txt
grep "step:  5" llep_memory_no_llep.txt
```

### Unit Tests

```bash
# LPT planning + SwiGLU FFN (5 tests, no GPU required)
python -m pytest tests/unit_tests/test_llep.py -v

# Grouped MM, Triton kernels, numerical correctness (17 tests, 1 GPU)
python -m pytest tests/unit_tests/test_llep_correctness.py -v

# Hook-based flow (59 tests, requires >= 2 GPUs)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py
```

## Files

| File | Description |
|------|-------------|
| `torchtitan/distributed/llep.py` | Main LLEP implementation (planning, dispatch, compute, combine) |
| `torchtitan/distributed/llep_kernels.py` | Triton kernels (fused_silu_gate, pad/unpad, assign_tokens, send_matrix) |
| `torchtitan/models/moe/moe.py` | MoE module with `LLEPConfig` and hook integration points |

## Performance Optimizations

The implementation includes several optimizations over the initial port (see `docs/llep_optimization_report_pr008.md` for details):

| Optimization | Speedup | Technique |
|-------------|---------|-----------|
| Triton pad/unpad | 4.6x / 6.5x | Row-parallel kernels (1 program per row) |
| Triton fused_silu_gate | 1.9x | Fused `silu(x1) * x3` in single pass |
| Triton assign_tokens | 3.6x | Numpy plan encoding + GPU kernel |
| Vectorized send_matrix | 1.9x | Numpy `add.at` + per-expert vectorized overlap |
| Selective weight packing | - | Avoids `torch.where` double-materialization |

**E2E Result:** +10.6% mean TPS (1470 -> 1625) on mini_kimi_k2_llep_ep8, 8xB200.

## Testing

All tests run via `torchrun` and require at least 2 GPUs:

```bash
# Unit tests (LPT planning, FFN, no GPU required)
python tests/unit_tests/test_llep.py

# Optimization correctness (grouped MM, Triton kernels, numerical, 17 tests)
torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py

# Hook-based flow comprehensive tests (59 tests, requires >= 2 GPUs)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py

# Run specific category
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --category topk

# List all tests
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --list
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_llep.py` | 5 | LPT planning, SwiGLU FFN (single-process) |
| `test_llep_correctness.py` | 17 | Grouped MM vs for-loop, Triton fused_silu_gate, numerical stability, benchmarks |
| `test_llep_hooks.py` | 59 | Hook-based flow (`dispatch` -> `compute` -> `combine`) across top_k, hyperparams, dimensions, backward, edge cases |
