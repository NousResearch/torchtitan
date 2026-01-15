# Batch Size Scheduler

Dynamic batch size scheduling for torchtitan, enabling batch size warmup/rampup during training.

## Overview

Batch size warmup is a technique used by large-scale training runs (DeepSeek-V3, Megatron, etc.) to improve training stability and efficiency. Instead of starting with the full batch size, training begins with a smaller batch size and gradually increases it.

**Key Design Principle**: The batch size scheduler is **completely orthogonal** to data stages and LR scheduler:

```
BatchSizeScheduler:  f(consumed_samples) → batch_size
DataStageManager:    f(current_step) → dataloader
LRScheduler:         f(current_step) → learning_rate
```

They don't know about each other. The training loop coordinates them independently.

## Schedule Modes

### 1. Constant (Default)

Fixed batch size throughout training. This is the default behavior and maintains backward compatibility with existing configs.

```toml
[batch_size_scheduler]
mode = "constant"
# Or simply omit the [batch_size_scheduler] section entirely
```

### 2. Linear Rampup

Smooth interpolation from `start_batch_size` to `global_batch_size` over `rampup_samples`.

**Used by**: DeepSeek-V3 (3072 → 15360 over 469B tokens)

```toml
[batch_size_scheduler]
mode = "linear"
start_batch_size = 1024
rampup_samples = 1000000000  # 1B samples

[training]
global_batch_size = 4096  # Target batch size
```

**Behavior**:
```
samples:     0 -------- 500M -------- 1B -------- 2B
batch_size:  1024 ----- 2560 ------- 4096 ------ 4096
                  (linear interpolation)  (constant)
```

### 3. Increment Rampup

Step-wise increments at regular intervals (Megatron style).

**Used by**: Megatron-LM

```toml
[batch_size_scheduler]
mode = "increment"
start_batch_size = 1024
increment = 1024
rampup_samples = 1000000000

[training]
global_batch_size = 4096
```

**Behavior**:
```
samples:     0 ---- 333M ---- 666M ---- 1B ---- 2B
batch_size:  1024   2048     3072     4096    4096
               (step increases)        (constant)
```

## Configuration Reference

```toml
[batch_size_scheduler]
mode = "constant"           # "constant", "linear", or "increment"
start_batch_size = 0        # Starting batch size (0 = use global_batch_size)
rampup_samples = 0          # Samples over which to ramp up (0 = no rampup)
increment = 0               # For "increment" mode (0 = auto, uses start_batch_size)
```

### Constraints

- `start_batch_size` must be divisible by `local_batch_size × data_parallel_degree`
- `global_batch_size` must be divisible by `local_batch_size × data_parallel_degree`
- `increment` must be divisible by `local_batch_size × data_parallel_degree`

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
│                                                              │
│  consumed_samples ──► BatchSizeManager ──► gradient_accum   │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                      │
│                    │   Scheduler     │                      │
│                    │  (stateless)    │                      │
│                    └─────────────────┘                      │
│                              │                               │
│            ┌─────────────────┼─────────────────┐            │
│            ▼                 ▼                 ▼            │
│     ConstantBatchSize  LinearRampup  IncrementRampup       │
└─────────────────────────────────────────────────────────────┘
```

## Checkpointing

The scheduler is **stateless** - only `consumed_samples` needs to be checkpointed:

```python
# Saved in checkpoint
{
    "step": 1000,
    "ntokens_seen": 4194304000,
    "consumed_samples": 1024000,  # ← This is all the scheduler needs
}
```

On resume, the scheduler automatically computes the correct batch size from `consumed_samples`.

**Backward Compatibility**: Old checkpoints without `consumed_samples` default to 0.

## Example Configurations

### DeepSeek-V3 Style

```toml
[batch_size_scheduler]
mode = "linear"
start_batch_size = 3072
rampup_samples = 114746093750  # 469B tokens / 4096 seq_len

[training]
global_batch_size = 15360
local_batch_size = 4
seq_len = 4096
```

### Megatron Style

```toml
[batch_size_scheduler]
mode = "increment"
start_batch_size = 1024
increment = 1024
rampup_samples = 244140625  # 1B tokens / 4096 seq_len

[training]
global_batch_size = 4096
local_batch_size = 4
seq_len = 4096
```

### Quick Test (Debug Model)

```toml
[batch_size_scheduler]
mode = "linear"
start_batch_size = 8
rampup_samples = 1000

[training]
global_batch_size = 32
local_batch_size = 4
steps = 100
```

## Logging

When batch size changes, you'll see:

```
[INFO] Batch size changed: 1024 -> 2048, grad_accum_steps=2
```

At initialization:

```
[INFO] Batch size scheduler: linear rampup 1024 -> 4096 over 1000000 samples
```
