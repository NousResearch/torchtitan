# CPU Offload Engine for MoE Training

Async CPU offload engine for large-scale Mixture-of-Experts training. Moves activations and weights between GPU and pinned CPU memory using dedicated CUDA streams. Forked from NVIDIA Megatron-LM's fine-grained activation offloading, refactored to be framework-agnostic.

## What It Offloads

| Type | Method | Memory Saved | TPS Cost | Status |
|---|---|---|---|---|
| **Expert activations** | `save_on_cpu` — save autograd tensors to pinned CPU | 36 GiB (23%) | 83% | Working |
| **Expert activations** | `checkpoint` — recompute expert forward in backward | 25 GiB (16%) | 12-15% | Working |
| **Expert weights** | `offload_weights()` — async D2H after forward, H2D before backward | 50 GiB (step 1) | WIP | Partial (step 1 works, step 2 OOM) |
| **Params+optimizer** | FSDP `CPUOffloadPolicy` (PyTorch built-in) | 55 GiB (35%) | 93% | Working (slow) |
| **All combined** | Activation + FSDP | 93 GiB (60%) | 93% | Working |

Benchmarked on Qwen3 30B-A3B, 128 experts, EP=8, batch=2, seq=4096, 8xB200.

## Configuration

```toml
[training]
# Activation offloading — offload expert intermediate activations
enable_activation_offload = true
activation_offload_mode = "checkpoint"   # "checkpoint" or "save_on_cpu"

# Weight offloading — offload expert weights between forward/backward
enable_weight_offload = false            # experimental

# FSDP parameter offloading (PyTorch built-in)
enable_cpu_offload = false
```

## How It Works

### Activation Offloading

In each MoE layer's expert forward, the engine either:

**`checkpoint` mode** (recommended): Uses `torch.utils.checkpoint` to recompute expert forward during backward instead of saving activations. No CPU involvement, trades compute for memory.

**`save_on_cpu` mode**: Uses `torch.autograd.graph.save_on_cpu(pin_memory=True)` to save all tensors that autograd would keep for backward to pinned CPU memory instead of GPU. Reloaded automatically during backward.

```python
# In MoE.forward() — clean, no manual stream management:
if self.offload_expert_fc1:
    routed_output = _expert_forward_with_offload(
        routed_input, num_tokens_per_expert, self.experts, self
    )
else:
    routed_output = self.experts(routed_input, num_tokens_per_expert)
```

### Weight Offloading (Experimental)

Registers post-forward and pre-backward hooks via `offload_weights()`:

```python
# In train.py — one line per expert module:
from torchtitan.distributed.cpu_offload import offload_weights
offload_weights(module.experts)
```

The engine automatically:
1. After expert forward: async D2H weights to pinned CPU, free GPU storage
2. Before expert backward: async H2D reload from CPU, restore storage
3. D2H overlaps with next-layer attention forward
4. H2D overlaps with next-layer attention backward

**Known limitation**: With FSDP, `resize_(0)` on DTensor storage races with NCCL internal streams. Step 1 shows 50 GiB freed (167→117 GiB), but crashes on step 2.

## Architecture

```
torchtitan/distributed/cpu_offload/
├── tensor_pool.py        — Pinned CPU memory pool, O(1) reuse after warmup
├── tensor_offloader.py   — General-purpose async D2H/H2D (dedicated CUDA streams)
├── offload_group.py      — Tensor batch with CUDA event sync
├── chunk_handler.py      — Core D2H/H2D engine (Megatron-style)
├── offload_manager.py    — Singleton orchestrator (VP/PP support)
├── autograd_hooks.py     — ActivationOffloadContext (Megatron's off_interface)
├── offload_api.py        — Clean API: offload_activation, offload_commit, offload_weights
├── hybrid_optimizer.py   — GPU/CPU split optimizer
├── utils.py              — Debug helpers, is_graph_capturing, summary printer
└── megatron_fork/        — Original Megatron code for reference
```

## Clean API

### Activation offloading (Megatron-style)

```python
from torchtitan.distributed.cpu_offload import offload_activation, offload_commit

# Wrap expert computation — engine handles D2H/H2D automatically
with offload_activation(should_offload, input_tensor, "expert_fc1") as x:
    output = self.linear(x)
output = offload_commit(output, "expert_fc1", release=[input_tensor])
```

### Weight offloading

```python
from torchtitan.distributed.cpu_offload import offload_weights

# Register once — hooks handle everything automatically
offload_weights(model.experts)
```

### TensorOffloader (low-level)

```python
from torchtitan.distributed.cpu_offload import TensorOffloader

offloader = TensorOffloader(pin_memory=True)
handle = offloader.offload(gpu_tensor, release_storage=True)  # async D2H
gpu_tensor = offloader.reload(handle)                          # async H2D
offloader.sync_reload()                                        # wait for H2D
```

## Benchmark Results

### Qwen3 30B-A3B, EP=8, batch=2, seq=4096, 8xB200, no activation checkpointing

| # | Config | Memory | TPS | What offloaded |
|---|---|---|---|---|
| 1 | Baseline | 154 GiB | 7,426 | Nothing |
| 2 | `save_on_cpu` | 118 GiB | 1,262 | Expert activations → pinned CPU |
| 3 | `checkpoint` | 129 GiB | 6,518 | Expert activations recomputed |
| 4 | FSDP offload | 100 GiB | 518 | Params + optimizer + grads → CPU |
| 5 | `save_on_cpu` + FSDP | 61 GiB | 318 | Everything → CPU |

### Qwen3 10B-A1B, EP=8, batch=5, seq=4096, 8xB200

| Config | Memory | TPS | Delta |
|---|---|---|---|
| Baseline | 166 GiB | 16,439 | — |
| `checkpoint` | 132 GiB | 14,299 | -20% mem, -13% TPS |
| FSDP offload | 154 GiB | 3,668 | -8% mem, -78% TPS |

## Tests

136 tests across 8 files, all passing:

```bash
pytest tests/unit_tests/cpu_offload/ -v
```

Covers: tensor pool, async D2H/H2D overlap, bitwise integrity (up to 20GB), CUDA events, MoE forward/backward correctness, gradient accumulation, torch.compile, CUDA graphs, multiple dtypes (f32/f16/bf16).
