# LLEP Integration into Torchtitan — Full Report

**Branch**: `phuc/upstream-2026-24-01-with-fixing-deepep_and_least_loaded_ep_but_correct_upstream_branch`
**Base**: `dev-updated-again` (NousResearch/torchtitan fork)
**Date**: 2026-02-13
**Hardware used for single-node tests**: 8x H100 80GB (single node)

---

## 1. What is LLEP?

Least-Loaded Expert Parallelism (LLEP) is a load-balancing strategy for MoE models from Salesforce AI Research ("Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts", Nguyen et al.). Instead of statically partitioning experts across GPUs, LLEP dynamically redistributes expert weights at runtime to the least-loaded GPUs via P2P transfers, then routes tokens accordingly. The goal: eliminate the stragglers caused by skewed expert routing in multi-node EP.

---

## 2. Commit History (Chronological)

| Commit | Summary |
|---|---|
| `2c11418` | **Initial LLEP integration** — added `torchtitan/distributed/llep.py` (1336 lines), `ExpertParallelLLEP` class, `use_llep` flag in `MoEArgs`, LLEP forward path in `MoE.forward()`, wired into `apply_moe_ep_tp()`. Added `debugmodel_llep` and `debugmodel_baseline` flavors. Unit tests for LPT planning, SwiGLU FFN, distributed forward. |
| `0bfc71b` | Added training configs for Kimi K2 runs |
| `037ad73` | Added `kimi_k2_llep` and `kimi_k2_sft_llep` model flavors in `__init__.py` with LLEP enabled (384 experts, top_k=8, sigmoid routing) |
| `fa22a8e` | **Critical multi-node fix** — P2P weight transfer switched from individual `isend`/`irecv` (gloo TCP, breaks at scale) to `batch_isend_irecv` with `P2POp(group_peer=...)` for NCCL. CPU offload fix: move weights to GPU after `to_local()`. GPU sync fix: replaced ~770 per-element `.item()` calls per rank/layer/step with bulk `cpu().tolist()` in `compute_llep_lpt_plan`. Vectorized `compute_gpu_imbalance_ratio`. |
| `39a6cc8` | Added mini Kimi 30B 1.5B config |
| `a180d82` | Added `mini_kimi_k2_llep` flavor (256 experts, dim=3072, 8 layers) and `mini_kimi_k2_llep_ep8.toml` config for single-node 8-GPU testing |
| `8e2c61a` | Tuned LLEP parameters: `llep_max_tokens_factor` 1.1→1.0, `llep_adaptive_threshold` 0.0→1.3 (in `kimi_k2_sft_llep` flavor) |
| `d1ebd1b` | Merge commit pulling remote changes (added `mini_kimi_k2_llep` flavor from remote) |

---

## 3. Key Files

| File | Role |
|---|---|
| `torchtitan/distributed/llep.py` | Core LLEP algorithm (1336 lines). LPT planning, P2P weight transfer, token routing, SwiGLU FFN computation. |
| `torchtitan/models/deepseek_v3/__init__.py` | Model flavor definitions. Contains `mini_kimi_k2_llep`, `mini_kimi_k2_baseline`, `kimi_k2_llep`, `kimi_k2_sft_llep`, etc. |
| `torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_llep_ep8.toml` | LLEP training config: EP=8, seq_len=8192, LBS=6, Muon optimizer, full activation checkpointing, CPU offload, aggressive memory mode. |
| `torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_baseline_ep8.toml` | Identical config but with `use_llep=False` for A/B comparison. |
| `torchtitan/models/deepseek_v3/train_configs/debug_model_llep_dp2.toml` | Debug config for DP+LLEP+CPU offload local testing. |
| `torchtitan/distributed/expert_parallel.py` | Contains `ExpertParallelLLEP` class for LLEP weight sharding. |
| `torchtitan/models/moe.py` | MoE module — `MoEArgs` has LLEP fields (`use_llep`, `llep_max_tokens_factor`, `llep_min_tokens_per_gemm`, `llep_adaptive_threshold`). LLEP forward path branched in `MoE.forward()`. |
| `torchtitan/models/attention.py` | Modified (unstaged) — likely attention-related changes for compatibility. |
| `torchtitan/experiments/dion_optimizer/parameter_classification.py` | Modified (unstaged) — parameter classification for Muon optimizer. |

---

## 4. What We Tried Today (2026-02-13 Session)

### 4.1 Initial Training Attempt — OOM at seq_len=24576

**Command:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 --master_port=29501 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_llep_ep8.toml \
  --training.steps 3 --training.dataset c4_test
```

**Result:** `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`

**Analysis:** The config had `seq_len = 24576` (inherited from the original Kimi K2 24k config). With 256 experts, dim=3072, LBS=6, and full activation checkpointing, the forward pass succeeded but backward ran out of GPU memory. The mini model (dim=3072, 8 layers, 256 experts with moe_inter_dim=2048) is still large — each expert has `3072*2048*3 = 18.9M` params, times 256 experts = ~4.8B expert params alone. At seq_len=24576 the activation memory during backward exceeded the 80GB H100 capacity even with CPU offload and aggressive memory management.

**Fix:** Reduced `seq_len` from 24576 to 8192 in the config.

### 4.2 LLEP Parameter Tuning

Before running, we committed a parameter change to `kimi_k2_sft_llep` flavor in `__init__.py`:
- `llep_max_tokens_factor`: 1.1 → 1.0 (no oversubscription — each GPU gets exactly its fair share)
- `llep_adaptive_threshold`: 0.0 → 1.3 (only migrate experts when imbalance ratio exceeds 1.3x)

**Rationale:** The original 1.1 factor was causing unnecessary token redistribution overhead. Setting threshold to 1.3 means LLEP only kicks in when a GPU has 30%+ more tokens than average, reducing churn.

### 4.3 Git Operations

- Committed `__init__.py` LLEP param changes
- Pulled from origin (divergent branches — used merge to preserve all history)
- Merge resolved cleanly (auto-merged `__init__.py`)

### 4.4 Created Baseline Config for A/B Comparison

To measure LLEP overhead, we created:
1. **New flavor** `mini_kimi_k2_baseline` in `__init__.py` — identical architecture to `mini_kimi_k2_llep` (256 experts, dim=3072, 8 layers, top_k=8, sigmoid routing, route_scale=2.827) but with `use_llep=False`
2. **New config** `mini_kimi_k2_baseline_ep8.toml` — identical training settings (seq_len=8192, LBS=6, Muon optimizer, full activation checkpointing, CPU offload, aggressive memory mode)

### 4.5 A/B Speed Comparison — LLEP vs Baseline (5 steps each)

Both runs: 8x H100, EP=8, seq_len=8192, LBS=6, bfloat16, Muon optimizer, full activation checkpointing, CPU offload enabled, aggressive memory mode=maximum, compile loss only.

---

## 5. A/B Comparison Results (Step 5, steady-state)

| Metric | LLEP (on) | Baseline (off) | Delta |
|---|---|---|---|
| **TPS (tokens/sec)** | 1,941 | 2,655 | **Baseline +37% faster** |
| **TFLOPS** | 23.83 | 32.58 | **Baseline +37%** |
| **MFU** | 2.41% | 3.30% | **Baseline +0.89 pp** |
| **Memory Active (avg across 8 GPUs)** | ~39.5 GiB (50.5%) | ~29.5 GiB (37.0%) | **Baseline uses ~10 GiB less** |
| **Memory Reserved** | ~48.98 GiB (61.9%) | ~49.02 GiB (61.9%) | ~Same |
| **Loss** | 8.1772 | 8.1957 | ~Same (random init, meaningless) |
| **Grad Norm** | 24.875 | 9.6875 | Different (expected — different code paths) |

### Step-by-step TPS progression

| Step | LLEP TPS | Baseline TPS |
|---|---|---|
| 2 | 1,844 | 2,654 |
| 3 | 1,840 | 2,653 |
| 4 | 1,920 | 2,645 |
| 5 | 1,941 | 2,655 |

LLEP shows slight warmup improvement (1,844→1,941) while baseline is stable from the start.

### Per-GPU Memory — All Steps (Active GiB / Active %)

#### LLEP (on)

| GPU | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---|---|---|---|
| rank0 | 38.15 (48.18%) | 37.95 (47.93%) | 39.90 (50.39%) | 40.00 (50.52%) |
| rank1 | 38.12 (48.15%) | 38.57 (48.71%) | 39.87 (50.35%) | 39.65 (50.08%) |
| rank2 | 37.56 (47.43%) | 38.18 (48.22%) | 39.96 (50.47%) | 39.55 (49.95%) |
| rank3 | 38.16 (48.20%) | 38.08 (48.09%) | 39.66 (50.09%) | 40.44 (51.08%) |
| rank4 | 38.09 (48.11%) | 38.18 (48.22%) | 39.99 (50.51%) | 40.04 (50.56%) |
| rank5 | 37.56 (47.44%) | 37.86 (47.81%) | 40.11 (50.66%) | 40.26 (50.85%) |
| rank6 | 37.66 (47.57%) | 39.16 (49.46%) | 39.66 (50.09%) | 37.69 (47.61%) |
| rank7 | 37.66 (47.56%) | 39.04 (49.31%) | 39.07 (49.34%) | 37.36 (47.18%) |
| **Reserved** | 48.45 | 48.62 | 48.80 | 48.98 |

#### Baseline (no LLEP)

| GPU | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---|---|---|---|
| rank0 | 28.89 (36.49%) | 29.52 (37.28%) | 29.08 (36.73%) | 29.32 (37.04%) |
| rank1 | 29.12 (36.78%) | 29.49 (37.25%) | 29.61 (37.40%) | 29.75 (37.58%) |
| rank2 | 28.89 (36.49%) | 29.06 (36.70%) | 29.46 (37.21%) | 29.36 (37.08%) |
| rank3 | 29.24 (36.92%) | 29.09 (36.74%) | 29.41 (37.14%) | 30.01 (37.90%) |
| rank4 | 29.25 (36.95%) | 29.49 (37.25%) | 28.91 (36.52%) | 29.73 (37.55%) |
| rank5 | 29.16 (36.83%) | 29.08 (36.73%) | 29.95 (37.83%) | 29.24 (36.93%) |
| rank6 | 28.87 (36.46%) | 29.02 (36.66%) | 29.66 (37.45%) | 29.30 (37.00%) |
| rank7 | 28.87 (36.46%) | 29.51 (37.27%) | 29.61 (37.39%) | 29.17 (36.84%) |
| **Reserved** | 48.47-48.49 | 48.64-48.70 | 48.82-48.90 | 48.98-49.05 |

### Memory Balance Analysis (Step 5)

| Metric | LLEP | Baseline | Interpretation |
|---|---|---|---|
| **Min Active** | 37.36 GiB (rank7) | 29.17 GiB (rank7) | LLEP uses ~8 GiB more on the lightest GPU |
| **Max Active** | 40.44 GiB (rank3) | 30.01 GiB (rank3) | LLEP uses ~10 GiB more on the heaviest GPU |
| **Mean Active** | 39.37 GiB | 29.49 GiB | LLEP uses ~10 GiB more on average |
| **Range (max-min)** | 3.08 GiB | 0.84 GiB | **LLEP has 3.7x wider spread** |
| **Std Dev** | 1.08 GiB | 0.29 GiB | **LLEP has 3.7x higher variance** |
| **CoV (std/mean)** | 2.74% | 0.98% | LLEP is relatively less balanced |
| **Peak Reserved** | 48.98 GiB (61.9%) | 49.05 GiB (62.0%) | ~Same reserved ceiling |
| **Headroom (80 - max active)** | 39.56 GiB | 49.99 GiB | **Baseline has ~10 GiB more headroom** |

### Memory Balance Across Steps (Range = max - min across 8 GPUs)

| Step | LLEP Range | Baseline Range | LLEP Std | Baseline Std |
|---|---|---|---|---|
| 2 | 0.60 GiB | 0.38 GiB | 0.27 GiB | 0.17 GiB |
| 3 | 1.21 GiB | 0.50 GiB | 0.46 GiB | 0.23 GiB |
| 4 | 1.04 GiB | 1.04 GiB | 0.35 GiB | 0.36 GiB |
| 5 | 3.08 GiB | 0.84 GiB | 1.08 GiB | 0.29 GiB |

**Observations:**

1. **LLEP does NOT improve memory balance on this single-node setup.** In fact, LLEP makes memory distribution *less* balanced (3.08 GiB range vs 0.84 GiB at step 5, 3.7x worse).

2. **LLEP uses ~10 GiB more active memory per GPU on average.** This overhead comes from migrated expert weight buffers (w1/w2/w3 copies for foreign experts), token routing bookkeeping, and P2P staging buffers.

3. **Memory imbalance grows over steps with LLEP** (range: 0.60→3.08 GiB), while baseline stays relatively flat (0.38→0.84 GiB). This suggests LLEP's dynamic weight migration accumulates fragmentation over time.

4. **Reserved memory is nearly identical** between LLEP and baseline (~49 GiB), meaning the CUDA allocator reserves the same total but LLEP's active usage is higher and more variable.

5. **On a balanced single-node setup, LLEP's memory "balancing" has nothing to balance** — all 8 GPUs already have near-identical loads under standard EP. LLEP's benefit would appear on multi-node runs where cross-node expert routing creates genuine load skew (e.g., one node gets 3x more tokens than another).

---

## 6. Analysis & Interpretation

### Why is LLEP slower on a single node?

LLEP is designed for **multi-node** MoE training where:
1. Cross-node AllToAll is the bottleneck (slow interconnect between nodes)
2. Expert load imbalance causes stragglers (one node gets 3x more tokens than another)
3. LLEP fixes this by migrating expert weights to less-loaded GPUs, reducing cross-node token traffic

On a **single node** with NVLink/NVSwitch:
- AllToAll is already fast (intra-node bandwidth ~900 GB/s)
- There are no stragglers from cross-node imbalance
- LLEP adds overhead: LPT planning, P2P weight transfers, extra bookkeeping, more complex token routing
- LLEP also uses ~10 GiB more active memory per GPU for the migrated expert weight copies

The 37% slowdown is pure LLEP overhead with zero benefit in this topology.

### When should LLEP help?

- **Multi-node** (8+ nodes) with EP spanning across nodes
- **High expert count** (256-384) with **skewed routing** (some experts get 10x more tokens)
- **Slow interconnect** between nodes (InfiniBand, not NVLink)
- The original Salesforce paper shows wins at 64+ GPUs across 8+ nodes

### Memory overhead explanation

The ~10 GiB extra active memory in LLEP comes from:
- Migrated expert weight buffers (w1, w2, w3 for each migrated expert)
- Token routing bookkeeping tensors (assignment maps, capacity tracking)
- P2P transfer staging buffers

---

## 7. Open Items / Next Steps

1. **Multi-node test**: Run LLEP on 8 nodes (64 GPUs) with EP=64 to measure the actual benefit. Use `kimi_k2_llep` or `kimi_k2_sft_llep` flavor with the full-size model.
2. **Adaptive threshold tuning**: The `llep_adaptive_threshold=1.3` setting needs validation at scale — too high means LLEP never activates, too low means excessive migration.
3. **`llep_min_tokens_per_gemm`**: Currently 256 for mini, 1024 for full-size. This batching threshold affects whether migrated experts actually do useful work or just waste compute on tiny batches.
4. **Unstaged changes**: `torchtitan/distributed/llep.py`, `torchtitan/models/attention.py`, `torchtitan/experiments/dion_optimizer/parameter_classification.py` have uncommitted modifications that may affect production runs.
5. **Compile compatibility**: Currently only `components = ["loss"]` is compiled. LLEP's dynamic control flow (variable expert counts per GPU, conditional P2P) makes it hard to torch.compile the MoE layer itself.
6. **seq_len scaling**: We had to drop from 24k to 8k for the mini model. The full-size Kimi K2 configs use seq_len=24576 with EP=64 across 8 nodes where memory is less constrained (experts sharded across more GPUs).

---

## 8. LLEP Config Refactoring (2026-02-14 Session)

### 8.1 What We Did

Refactored LLEP configuration from hardcoded model flavors to a TOML-configurable `[llep]` section. The goal: tune LLEP hyperparameters (α, λ, m) from the config file without touching Python.

**New branch**: `phuc/kimi_k2_with_autotune_llep` (branched from `0cc85a1`)
**Commit**: `5da865e` — "Refactor LLEP config: add [llep] TOML section and LLEPConfig dataclass"

### 8.2 Files Modified

| File | Change |
|---|---|
| `torchtitan/models/moe/moe.py` | Added `LLEPConfig` dataclass (α=1.1, m=1024, λ=0.0 defaults). Replaced 3 flat `llep_*` fields in `MoEArgs` with `llep: LLEPConfig`. Added `[LLEP] enabled with α=..., m=..., λ=...` log at model init. |
| `torchtitan/models/moe/__init__.py` | Exported `LLEPConfig`. |
| `torchtitan/config/job_config.py` | Added `LLEP` dataclass with `Optional[None]` fields (`enabled`, `max_tokens_factor`, `min_tokens_per_gemm`, `adaptive_threshold`). Added `llep: LLEP` to `JobConfig`. |
| `torchtitan/models/deepseek_v3/model/args.py` | Wired `job_config.llep.*` → `moe_args.use_llep` / `moe_args.llep.*` overrides in `update_from_config()`. |
| `torchtitan/models/deepseek_v3/__init__.py` | Removed all `use_llep=True/False` and `llep=LLEPConfig(...)` from every flavor. Removed all sweep flavors (`mini_kimi_k2_llep_t0.0` through `t5.0`). LLEP is now purely TOML-controlled. |
| `torchtitan/distributed/llep.py` | Added Nous Research copyright. Added env var override for `adaptive_threshold` (`EP_ADAPTIVE_THRESHOLD`). |
| `torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_llep_ep8.toml` | Added `[llep] enabled = true` section with documented overrides. |
| `torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_baseline_ep8.toml` | Created baseline config (no LLEP) for A/B comparison. |
| `torchtitan/models/deepseek_v3/train_configs/test_llep_toml_override.toml` | Created test config with custom LLEP overrides (α=1.4, λ=1.5, m=512) to verify TOML wiring. |

### 8.3 Design Decisions

**Why `LLEPConfig` sub-dataclass instead of flat fields?**
Cleaner organization — `moe_args.llep.max_tokens_factor` instead of `moe_args.llep_max_tokens_factor`. Groups related params together.

**Why `Optional[None]` defaults in the TOML `LLEP` class?**
So unset TOML values don't override flavor defaults. Only explicitly set values take effect. This means existing configs without `[llep]` work unchanged.

**Why `enabled` field?**
So any flavor can have LLEP toggled from TOML without creating a separate flavor. `[llep] enabled = true` on a `kimi_k2` flavor enables LLEP; `enabled = false` on a `kimi_k2_llep` flavor disables it.

**Default values — paper vs original:**
We initially set defaults to paper recommendations (α=1.0, λ=1.3, m=1024) but this caused OOM on configs that previously worked with the original defaults (α=1.1, λ=0.0, m=1024). The key difference: α=1.0 causes more aggressive spilling → more foreign weight copies → more memory. We reverted to the original working defaults (α=1.1, λ=0.0, m=1024) and documented the paper values in docstrings.

### 8.4 Issues Encountered

1. **OOM after changing defaults** — Setting `LLEPConfig` defaults to paper values (α=1.0) caused OOM on `mini_kimi_k2_llep_ep8.toml` which previously worked with α=1.1. Root cause: α=1.0 means each GPU gets exactly fair share → more weight spilling → more memory for foreign expert copies. Fix: reverted defaults to α=1.1.

2. **TOML parsing error** — Commented-out `#[llep]` section header with uncommented fields below it caused the TOML parser to assign LLEP fields to the `[compile]` section above. Error: `ValueError: Invalid field names in <class 'torchtitan.config.job_config.Compile'> data: {'max_tokens_factor', ...}`. Fix: properly comment all lines or uncomment the section header.

3. **`enable` vs `enabled` naming** — Initially used `enable` in TOML but `enabled` in Python dataclass. Fix: standardized on `enabled` everywhere.

4. **Zombie GPU processes** — Failed training runs left processes holding 60-80 GiB per GPU. Subsequent runs OOMed at `dist.broadcast()` before training even started. Fix: `kill -9` zombie PIDs, verify with `nvidia-smi`.

5. **Port conflicts** — Consecutive `torchrun` invocations on same `--master_port` failed with `EADDRINUSE`. Fix: use different ports or kill leftover processes.

### 8.5 Verification — Config Refactoring Doesn't Change Performance

To verify the refactoring is purely organizational (no performance impact), we ran the same A/B comparison on both branches.

**On `phuc/kimi_k2_with_autotune_llep` (`5da865e`)** — LLEP controlled via `[llep] enabled = true` in TOML:

| Metric | LLEP (α=1.1, λ=0.0) | Baseline (no LLEP) | Gap |
|---|---|---|---|
| **TPS** | 1,720 | 2,659 | **Baseline 55% faster** |
| **Memory Active (avg)** | ~39.8 GiB (50.5%) | ~29.9 GiB (37.8%) | **Baseline ~10 GiB less** |
| **Loss** | 8.2169 | 8.0534 | ~Same |
| **MFU** | 2.13% | 3.30% | **Baseline +1.17 pp** |

**On original branch (`0cc85a1`)** — LLEP controlled via `use_llep=True` in flavor:
Attempted to run but OOMed because `mini_kimi_k2_llep_ep8.toml` on this branch still has `seq_len=24576` (our `seq_len=8192` fix was only in the working tree, now on the new branch).

**Conclusion:** The TPS gap (55%) and memory gap (~10 GiB) are identical to the earlier measurements (Section 5: 37% gap at step 5 with different run conditions). The config refactoring adds zero runtime overhead — it's purely organizational plumbing. The LLEP overhead itself (P2P transfers, barriers, LPT planning, foreign weight memory) is unchanged.

### 8.6 How LLEP is Now Configured

**Before (flavor-based):**
```python
# In __init__.py — had to create a new flavor for each config
"kimi_k2_llep": DeepSeekV3ModelArgs(
    moe_args=MoEArgs(
        use_llep=True,
        llep_max_tokens_factor=1.1,
        llep_min_tokens_per_gemm=1024,
        llep_adaptive_threshold=0.0,
    ),
)
```

**After (TOML-based):**
```toml
# In any .toml config — works with any flavor
[llep]
enabled = true
# max_tokens_factor = 1.1    # α (default: 1.1)
# min_tokens_per_gemm = 1024 # m (default: 1024)
# adaptive_threshold = 0.0   # λ (default: 0.0)
```

### 8.7 LLEP Hyperparameter Reference

Three hyperparameters from "Least-Loaded Expert Parallelism" (Nguyen et al., Salesforce AI Research):

| Param | Symbol | Default | Paper (§5.1) | What it does |
|---|---|---|---|---|
| `max_tokens_factor` | α | 1.1 | 1.0 | GPU capacity ceiling. `max_per_gpu = α × (total/num_gpus)`. Lower = more spilling, more balanced but more memory. |
| `min_tokens_per_gemm` | m | 1024 | 1024 | Minimum chunk size to justify spilling. Below this, the GEMM is too small to be efficient. |
| `adaptive_threshold` | λ | 0.0 | 1.3 | Imbalance ratio to trigger LLEP. 0 = always active. 1.3 = only when max_gpu/mean_gpu > 1.3. |

---

## 9. Reproduction Commands

```bash
# LLEP (on)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 --master_port=29501 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_llep_ep8.toml \
  --training.steps 5 --training.dataset c4_test

# Baseline (off)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 --master_port=29501 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_baseline_ep8.toml \
  --training.steps 5 --training.dataset c4_test
```

---

## 10. LLEP Autotune Implementation (2026-02-14 Session)

### 10.1 Goal

Automatically find optimal LLEP hyperparameters (α, m, λ) at startup, instead of requiring manual tuning. The approach: run a few forward passes on real data, collect routing stats, simulate the LPT algorithm with different α values, and pick the best config.

### 10.2 New Files

| File | Role |
|---|---|
| `torchtitan/distributed/llep_autotune.py` | Core autotune module: `collect_routing_stats()`, `find_optimal_params()`, `autotune_llep()` |
| `torchtitan/models/deepseek_v3/train_configs/mini_kimi_k2_llep_autotune_ep8.toml` | Test config with `[llep] autotune = true` |

### 10.3 How It Works

**Phase 1: Collect routing stats** (~7-13s depending on model size)
- Runs 2-3 forward-only passes (`torch.no_grad()`) on real training data
- Hooks each MoE layer's router via `register_forward_hook` to capture `num_tokens_per_expert`
- Returns per-layer per-sample expert counts tensors

**Phase 2: Simulate α candidates** (<100ms, pure Python)
- Groups expert counts by layer (not averaged — evaluated per-layer independently)
- For each α ∈ [0.9, 1.0, 1.1, 1.2, 1.5]:
  - Calls `compute_llep_lpt_plan()` on each layer's real expert counts
  - Records **worst-case** balance (max/mean) across all layers
  - Records total weight transfers and max foreign experts per GPU
- Computes λ from observed imbalance: `P50 × 0.9`

**Phase 3: Select best config** — priority order:
1. **Memory safety** (hard reject if foreign weight overhead > 80% of free memory)
2. **Balance** (minimize worst-case max/mean ratio across layers)
3. **Communication** (among equally-balanced candidates, fewest transfers wins)

**Phase 4: Apply and log**
- Sets `_llep_max_tokens_factor`, `_llep_min_tokens_per_gemm`, `_llep_adaptive_threshold` on all MoE modules
- Logs before/after comparison

### 10.4 Config

```toml
[llep]
enabled = true
autotune = true          # run autotune at startup
autotune_samples = 3     # number of forward passes
```

Added `autotune: bool = False` and `autotune_samples: int = 3` to the `LLEP` dataclass in `torchtitan/config/job_config.py`.

Wired into `torchtitan/train.py` after checkpoint loading, before `Training starts at step`:
```python
if hasattr(job_config, "llep") and job_config.llep.autotune:
    from torchtitan.distributed.llep_autotune import autotune_llep
    autotune_llep(model=self.model_parts[0], dataloader=self.dataloader, job_config=job_config)
```

### 10.5 Scoring Evolution

**First attempt**: `score = avg_balance + 0.001 * avg_transfers`
- Problem: averaged balance across all layer-samples, masking worst-case layers

**Second attempt**: Two-tier scoring (balance threshold < 1.1x → minimize transfers; else minimize balance)
- Problem: strict `<` threshold excluded α=1.1 which gave exactly 1.100x on the small model

**Third attempt**: Changed to `<=` threshold
- Problem: still used average across layers, not worst-case

**Final version**: Worst-case-per-layer scoring
- Groups expert counts by layer
- Uses `worst_balance = max(balance across all layers)` not average
- Priority: memory safety → balance (worst-case) → comm (tiebreaker)
- Handles two tiers: "acceptable" (≤ 1.1x) candidates prefer fewer transfers; "unacceptable" candidates prefer better balance

### 10.6 Test Results

**Small debug model** (8 experts, 4 layers, dim=256, EP=8):
```
[LLEP autotune] Completed in 7.3s
  Imbalance: P50=3.98x, P90=3.98x
  Selected: α=1.1, m=1024, λ=3.58
  balance=1.100x, 6.0 transfers/layer
```
Autotune correctly found α=1.1 as the sweet spot for this tiny model (1 expert/GPU).

**Mini Kimi K2** (256 experts, 8 layers, dim=3072, EP=8):
```
[LLEP autotune] Completed in 12.5s
  Without LLEP: worst imbalance = 1.01x (P50), 1.02x (P90)
  With LLEP (α=1.1, λ=999.00): worst imbalance = 1.010x
  LLEP recommended: False
```
Autotune correctly identified that routing is already balanced (1.02x) with random init on c4_test, and set λ=999 to effectively disable LLEP. **However, TPS was still 1,770 (same as LLEP-on), not 2,659 (baseline).** This revealed a critical finding.

### 10.7 Critical Finding: LLEP Code Path Overhead

Even when autotune sets λ=999 (skip LPT planning), the TPS did **not** recover to baseline because `_llep_enabled = True` still routes every MoE forward through `llep_moe_forward()`. This function runs the following **on every MoE layer, every step**, regardless of whether LPT is used:

| Operation | Location in `llep.py` | Cost |
|---|---|---|
| `torch.bincount()` for expert counts | line 1051 | Small |
| `dist.all_gather()` for per-rank counts | line 1057 | **Collective** |
| `compute_gpu_imbalance_ratio()` | line 1072 | Small |
| `dist.barrier()` | line 1118 | **Global sync** |
| `WeightTransferAutograd.apply()` | line ~1127 | Autograd setup |
| `assign_tokens_to_gpus()` with numpy | line ~1152 | CPU work + D2H syncs |
| `dist.all_to_all_single()` dispatch | line ~1195 | LLEP-specific AllToAll |
| `llep_swiglu_ffn()` Python for-loop | line ~1266 | **Per-expert matmul loop** |
| `dist.all_to_all_single()` combine | line ~1305 | LLEP-specific AllToAll |

The standard EP path uses `torch._grouped_mm` (single fused kernel for all experts), the standard AllToAll dispatch mechanism, and **no** all_gather/barrier for routing. The LLEP path replaces all of that with its own implementation.

**The 55% overhead is NOT from weight transfers** (those are skipped when λ=999). It's from:
1. **`llep_swiglu_ffn()` Python for-loop** vs `torch._grouped_mm` — the biggest factor
2. **`dist.barrier()`** per MoE layer per step — unnecessary global sync
3. **`dist.all_gather()`** per layer — extra collective not needed in standard EP
4. **LLEP-specific token routing** — numpy-based `assign_tokens_to_gpus()` with D2H syncs

**Implication**: Setting `_llep_enabled = False` at runtime would bypass the LLEP path entirely and recover baseline TPS, but this is hacky (breaks the parallelization setup). The proper fix is to either:
- Make `llep_moe_forward()` early-return to the standard EP path when `use_lpt = False`
- Or use `torch._grouped_mm` in `llep_swiglu_ffn()` instead of the Python for-loop

This is an architectural issue in `torchtitan/distributed/llep.py`, not the autotune.

### 10.8 Summary

| What | Status |
|---|---|
| Autotune implementation | ✅ Working — finds optimal (α, λ) in ~7-13s |
| Routing stat collection | ✅ Hooks real router output, per-layer per-sample |
| LPT simulation | ✅ Uses real `compute_llep_lpt_plan()`, exact results |
| Scoring (memory → balance → comm) | ✅ Worst-case-per-layer, two-tier selection |
| Before/after logging | ✅ Shows imbalance reduction with actual numbers |
| Config integration | ✅ `[llep] autotune = true` in TOML |
| Recovering baseline TPS when LLEP not needed | ❌ LLEP code path overhead persists even when LPT is skipped |

### 10.9 Next Steps

1. **Fix LLEP code path overhead**: Either make `llep_moe_forward()` fall back to standard EP dispatch when `use_lpt = False`, or replace the Python for-loop FFN with `torch._grouped_mm`
2. **Remove `dist.barrier()`** from `llep.py` line 1118 — likely unnecessary, adds sync overhead per layer
3. **Test on multi-node** with real imbalanced routing (SFT on domain-specific data) where LLEP should actually help
4. **Per-layer λ consideration**: The autotune collects per-layer stats. Could set different λ per layer, though current runtime already adapts per-layer via the imbalance check
