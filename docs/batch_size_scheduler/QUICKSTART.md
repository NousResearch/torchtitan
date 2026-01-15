# Batch Size Scheduler - Quick Start

## TL;DR

Add `[batch_size_scheduler]` to your config TOML:

```toml
# Linear rampup (recommended for large models)
[batch_size_scheduler]
mode = "linear"
start_batch_size = 1024      # Start small
rampup_samples = 1000000000  # Ramp over 1B samples

[training]
global_batch_size = 4096     # Target batch size
```

That's it! Old configs without `[batch_size_scheduler]` continue to work unchanged.

## Quick Reference

| Mode | Config | Behavior |
|------|--------|----------|
| **Constant** | `mode = "constant"` or omit section | Fixed batch size |
| **Linear** | `mode = "linear"` | Smooth ramp: start → end |
| **Increment** | `mode = "increment"` | Step-wise: start, start+inc, start+2*inc, ... |

## Formula

```
Linear:    batch_size = start + (consumed_samples / rampup_samples) * (end - start)
Increment: batch_size = start + floor(consumed_samples / samples_per_step) * increment
```

## Common Configurations

### DeepSeek-V3 Style
```toml
[batch_size_scheduler]
mode = "linear"
start_batch_size = 3072
rampup_samples = 114746093750  # 469B tokens / 4096 seq_len
```

### Megatron Style
```toml
[batch_size_scheduler]
mode = "increment"
start_batch_size = 1024
increment = 1024
rampup_samples = 244140625  # 1B tokens / 4096 seq_len
```

## Testing

```bash
# Unit tests
python docs/batch_size_scheduler/test_unit.py

# Integration tests
bash docs/batch_size_scheduler/test_batch_size_scheduler.sh
```
