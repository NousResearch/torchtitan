# Fine-Grained CPU Activation Offload Engine

A modular, framework-agnostic activation offloading engine for large-scale model training. Moves intermediate activations from GPU to pinned CPU memory during forward and reloads them during backward, using dedicated CUDA streams for fully asynchronous, overlapped transfers.

Only dependency is PyTorch.

## Why This Exists

PyTorch's built-in `FSDP2 CPUOffloadPolicy` offloads **parameters and optimizer states** only. For MoE models, **activations dominate memory** (e.g., DeepSeek-V3: 131GB activations vs 32GB params+optimizer). This engine offloads activations with per-module granularity — offload the expensive modules (expert FC layers, attention), recompute the cheap ones (LayerNorm).

| | FSDP2 CPUOffloadPolicy | This Engine |
|---|---|---|
| What's offloaded | Parameters, optimizer states | **Activations** |
| Granularity | All-or-nothing per FSDP unit | Per-module selective |
| Async streams | Internal, no control | 2 dedicated (D2H + H2D) |
| Memory pool | None | Pinned tensor pool, O(1) reuse |
| Autograd integration | None | Intercepts `save_for_backward` |

## Quick Start

```python
from torchtitan.distributed.cpu_offload import (
    ActivationOffloadContext,
    OffloadManager,
    group_commit,
)

# At the start of each microbatch:
ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0)

# Wrap each module you want to offload:
with ActivationOffloadContext(True, input_tensor, "expert_fc1") as x:
    output = self.linear_fc1(x)
output = group_commit(output, "expert_fc1", forced_released_tensors=[input_tensor])

# At iteration boundary:
ActivationOffloadContext.reset()
```

### MoE Expert Example

```python
class ExpertMLP(nn.Module):
    def forward(self, x):
        # Offload fc1 input
        with ActivationOffloadContext(self.offload_fc1, x, "expert_fc1") as x_in:
            h = self.fc1(x_in)
        h = group_commit(h, "expert_fc1", forced_released_tensors=[x])

        # Offload activation output
        with ActivationOffloadContext(self.offload_act, h, "moe_act") as h_in:
            h = self.activation(h_in)
        h = group_commit(h, "moe_act", forced_released_tensors=[])

        return self.fc2(h)
```

### Optimizer State Offloading (Optional)

```python
from torchtitan.distributed.cpu_offload import HybridDeviceOptimizer

optimizer = HybridDeviceOptimizer(
    model.parameters(),
    cpu_optimizer_cls=torch.optim.AdamW,
    gpu_optimizer_cls=torch.optim.AdamW,
    offload_fraction=0.5,  # 50% of params to CPU
    overlap_cpu_optimizer_d2h_h2d=True,
)
```

## How It Works

### Forward Pass

```
Compute stream:  [Module 0] ─────── [Module 1] ─────── [Module 2] ───
D2H stream:                [D2H 0] ─────────── [D2H 1] ──────────────
                            ↑ overlapped         ↑ overlapped
```

After each module computes, its input activations are copied GPU→CPU on a dedicated D2H stream, running concurrently with the next module's compute.

### Backward Pass (Layer-Staggered Reload)

```
Compute stream:  [Backward L2] ───── [Backward L1] ───── [Backward L0]
H2D stream:       [Reload L1] ─────── [Reload L0] ────────────────────
                   ↑ overlapped         ↑ overlapped
```

Reloads layer N's activations from CPU while computing layer N+1's backward. Only one activation per module type on GPU at any time.

### Key Design Choices

- **Shared group names**: All transformer layers use the same name (e.g., `"expert_fc1"`). The engine only skips the *last* occurrence (nothing after it to hide behind). Earlier layers still offload.
- **Pinned memory pool**: After warmup, all D2H/H2D copies reuse pre-allocated pinned tensors — no `cudaMallocHost` overhead.
- **Autograd hooks**: `torch._C._autograd._push_saved_tensors_default_hooks` intercepts `save_for_backward` transparently. Tensors are replaced with CPU-backed references.
- **`forced_released_tensors`**: Explicitly calls `untyped_storage().resize_(0)` to free GPU memory immediately instead of waiting for Python GC.
- **CUDA events**: Per-group offload/reload events synchronize streams without blocking. Compatible with CUDA Graphs (uses external events, not stream sync).

## API Reference

### Activation Offloading

```python
# Initialize for a microbatch
ActivationOffloadContext.init_chunk_handler(
    vp_size=1,                       # Virtual pipeline size
    vp_stage=0,                      # Current VP stage
    min_offloaded_tensor_size=1024*1024,  # Skip small tensors (numel)
)

# Wrap a module's forward
with ActivationOffloadContext(offload: bool, tensor, name: str) as tensor:
    output = module(tensor)

# Mark end of group, trigger D2H
output = group_commit(
    output,
    name="expert_fc1",
    forced_released_tensors=[input],  # Free these immediately
    delay_offload=False,              # Defer to flush_delayed_groups()
)

# Mark parameters as never offloadable
ActivationOffloadContext.mark_not_offloadable(param)

# Reset between iterations
ActivationOffloadContext.reset()

# Temporarily disable/enable
disable_offload()
enable_offload()
```

### Optimizer Offloading

```python
HybridDeviceOptimizer(
    params,
    cpu_optimizer_cls=torch.optim.AdamW,
    gpu_optimizer_cls=torch.optim.AdamW,
    offload_fraction=0.5,             # 0.0 = all GPU, 1.0 = all CPU
    param_update_in_fp32=False,       # FP32 master weights
    pin_cpu_grads=True,               # Pin gradient buffers
    overlap_cpu_optimizer_d2h_h2d=True,  # Async streams
)
```

## Configuration Tips

- **Use shared group names** across layers (e.g., all layers use `"expert_fc1"`). This enables the layer-staggered reload pattern where only the last occurrence is skipped.
- **Offload expensive, recompute cheap**: Use offloading for attention and expert FC layers; use activation checkpointing for LayerNorm and activation functions.
- **`min_offloaded_tensor_size`**: Skip small tensors to avoid PCIe overhead exceeding the memory savings.
- **`forced_released_tensors`**: Pass the input tensor to free GPU memory immediately after it's been copied to CPU. Don't rely on Python GC.
