# Multi-Stage Data Training

Multi-stage training allows switching between different data mixtures at specified training steps, similar to approaches used in Qwen3, DeepSeek-V3, and Llama 3.

## Quick Start

Data stages are **optional**. If no `[[training.data_stages]]` are defined, a single stage is auto-created from `[training]` data fields (backward compatible). When stages ARE defined, they override `[training]` data fields completely.

### Multi-Stage Example

Define `[[training.data_stages]]` sections for multi-stage training:

```toml
[training]
steps = 150000

[[training.data_stages]]
name = "general"
start_step = 0
end_step = 100000
dataset_type = "nanoset"
dataset_folders = ["/data/general", "/data/math", "/data/code"]
dataset_weights = [0.8, 0.1, 0.1]
seq_len = 4096

[[training.data_stages]]
name = "reasoning"
start_step = 100000
end_step = 130000
dataset_type = "nanoset"
dataset_folders = ["/data/general", "/data/math", "/data/code"]
dataset_weights = [0.3, 0.35, 0.35]
seq_len = 4096

[[training.data_stages]]
name = "long_context"
start_step = 130000
dataset_type = "nanoset"
dataset_folders = ["/data/general", "/data/math", "/data/code"]
dataset_weights = [0.3, 0.35, 0.35]
seq_len = 32768
```

## Configuration Fields

Each `[[training.data_stages]]` section must define all data-related fields explicitly:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Stage identifier for logging |
| `start_step` | int | Yes | Step when stage begins (inclusive) |
| `end_step` | int | No | Step when stage ends (exclusive). Omit for final stage |
| `dataset` | string | Yes* | Dataset name (for huggingface type) |
| `dataset_path` | string | No | Path to dataset |
| `dataset_type` | string | Yes | `"huggingface"`, `"nanoset"`, `"preprocessed"`, `"packed_memmap"` |
| `dataset_folders` | list | Yes* | Folders for nanoset datasets |
| `dataset_weights` | list | No | Weights for blending datasets (must sum to 1.0) |
| `dataset_random_seed` | int | No | Random seed for this stage (defaults to `training.dataset_random_seed`) |
| `seq_len` | int | Yes | Sequence length |

*Required based on `dataset_type`: `dataset` for huggingface, `dataset_folders` for nanoset.

## Single-Stage Training (Backward Compatible)

For single-stage training, you can simply use `[training]` data fields - no `[[training.data_stages]]` needed:

```toml
[training]
steps = 100000
dataset_type = "huggingface"
dataset = "c4_test"
seq_len = 4096
```

A single stage named "default" is auto-created internally. This maintains full backward compatibility with existing configs.

Alternatively, you can explicitly define a single stage:

```toml
[training]
steps = 100000

[[training.data_stages]]
name = "pretrain"
start_step = 0
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books"]
dataset_weights = [0.7, 0.3]
seq_len = 4096
```

## Validation

The following validations are performed at startup:

- **Stage coverage**: First stage must start at step 0, no gaps or overlaps between stages
- **Required fields**: `name`, `start_step`, `dataset_type`, `seq_len` must be defined
- **Dataset source**: `dataset` required for huggingface, `dataset_folders` required for nanoset
- **Weights**: Must be non-negative, each <= 1.0, sum to 1.0, count must match folders
- **Value ranges**: `seq_len > 0`, `dataset_random_seed >= 0`, `start_step < training.steps`

## Common Patterns

### Pattern 1: Change Data Mixture

```toml
[[training.data_stages]]
name = "pretrain"
start_step = 0
end_step = 100000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books", "/data/code"]
dataset_weights = [0.7, 0.2, 0.1]  # 70% web, 20% books, 10% code
seq_len = 4096

[[training.data_stages]]
name = "annealing"
start_step = 100000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books", "/data/code"]
dataset_weights = [0.4, 0.3, 0.3]  # More balanced for final phase
seq_len = 4096
```

### Pattern 2: Context Extension

```toml
[[training.data_stages]]
name = "base"
start_step = 0
end_step = 90000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books", "/data/code"]
dataset_weights = [0.5, 0.3, 0.2]
seq_len = 4096

[[training.data_stages]]
name = "long_context"
start_step = 90000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books", "/data/code"]
dataset_weights = [0.5, 0.3, 0.2]
seq_len = 32768
```

### Pattern 3: Different Random Seeds (Multi-Epoch)

```toml
[[training.data_stages]]
name = "epoch1"
start_step = 0
end_step = 50000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books"]
dataset_weights = [0.7, 0.3]
dataset_random_seed = 1234
seq_len = 4096

[[training.data_stages]]
name = "epoch2"
start_step = 50000
dataset_type = "nanoset"
dataset_folders = ["/data/web", "/data/books"]
dataset_weights = [0.7, 0.3]
dataset_random_seed = 5678
seq_len = 4096
```

### Pattern 4: Mid-Training Ablation

For ablation studies where you want to test different data mixtures from a checkpoint, you can add stages that start mid-training. The system will auto-create a "default" stage from `[training]` fields for the gap.

**Ablation config** (start new mixture at step 5):
```toml
[training]
steps = 10
# These fields cover steps 0-5 (auto-created as "default")
dataset = "c4_test"
dataset_type = "huggingface"
seq_len = 512

# Ablation stage starts at step 5 with different random seed
[[training.data_stages]]
name = "ablation_stage"
start_step = 5
dataset = "c4_test"
dataset_type = "huggingface"
seq_len = 512
dataset_random_seed = 9999  # Different seed for ablation
```

The system auto-creates "default" for steps 0-5 from `[training]`, then transitions to "ablation_stage" at step 5.

## Logging

At training start, a stage plan is logged:

```
============================================================
DATA STAGE TRAINING PLAN
============================================================
Total stages: 3

Stage 1: general
  Steps: 0 -> 100,000 (100,000 steps)
  Estimated tokens: 409.60B tokens
  Dataset type: nanoset
  Dataset folders: 3 folders
  Weights: [0.800, 0.100, 0.100]
  Sequence length: 4096

Stage 2: reasoning
  Steps: 100,000 -> 130,000 (30,000 steps)
  ...
============================================================
```

At each transition:

```
============================================================
DATA STAGE TRANSITION
============================================================
Step 100000: 'general' -> 'reasoning'
Changes: dataset_weights
New weights: [0.300, 0.350, 0.350]
============================================================
```

## Checkpoint & Resume

Stage state is automatically saved in checkpoints:
- `stage_idx`: Current stage index
- `stage_name`: Current stage name
- `dataloader_state`: Position within the dataset

On resume, the exact stage and dataloader position are restored. No manual intervention needed.

## Testing

### Test Configs

Test configs are located in `docs/data_stages/configs/`:

| Config | Description |
|--------|-------------|
| `data_stages_test.toml` | 3 stages with transitions at step 5 and 10 |
| `data_stages_backcompat_test.toml` | No data_stages (backward compatibility) |
| `data_stages_ablation_test.toml` | Stages start at step 5 (ablation use case) |

### Automated Test Suite

Run the test script to verify all functionality:

```bash
./scripts/test_data_stages.sh
```

The test suite runs 5 tests:

```
============================================================
DATA STAGES TEST SUITE
============================================================

[Test 1] Backward Compatibility: No [[training.data_stages]]
  ✓ Auto-created 'default' stage from [training]
  ✓ Stage named 'default'
  ✓ Training completed successfully

[Test 2] Multi-Stage Training: Full run with 3 stages
  ✓ Transition at step 5: stage_1_general -> stage_2_reasoning
  ✓ Transition at step 10: stage_2_reasoning -> stage_3_final

[Test 3] Checkpoint Resume: from step 7
  ✓ Stage correctly restored to stage_2_reasoning
  ✓ Dataloader position restored
  ✓ Training resumed at correct step (8)

[Test 4] Reproducibility: Comparing losses between full and resumed runs
Step  | Full Run | Resume   | Match
------|----------|----------|------
8     | 4.7073   | 4.7073   | ✓
9     | 4.0312   | 4.0312   | ✓
10    | 4.0548   | 4.0548   | ✓
11    | 3.8143   | 3.8143   | ✓
12    | 3.8702   | 3.8702   | ✓
13    | 4.2306   | 4.2306   | ✓
14    | 3.6354   | 3.6354   | ✓
15    | 3.7099   | 3.7099   | ✓

[Test 5] Ablation: Stages start at step 5
  ✓ Auto-created 'default' stage for gap (steps 0-5)
  ✓ First stage is 'default'
  ✓ Second stage is 'ablation_stage'
  ✓ Transition occurred: default -> ablation_stage
  ✓ Training completed successfully

============================================================
ALL TESTS PASSED!
============================================================
```

### Manual Testing

```bash
# Backward compatibility (no data_stages)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file docs/data_stages/configs/data_stages_backcompat_test.toml

# Multi-stage training
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file docs/data_stages/configs/data_stages_test.toml

# Resume from step 7
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file docs/data_stages/configs/data_stages_test.toml \
    --checkpoint.load_step 7

# Ablation (stages start mid-training)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file docs/data_stages/configs/data_stages_ablation_test.toml
```

### What the Tests Verify

1. **Backward compatibility**: Existing configs without `[[training.data_stages]]` still work
2. **Stage transitions**: Dataloader rebuilds correctly at stage boundaries
3. **Checkpoint saves**: Stage index + exact dataloader position (sample count)
4. **Resume restores**: Exact state - losses match between full run and resumed run
5. **Ablation mode**: When `[[training.data_stages]]` starts after step 0 (e.g., step 5), the system auto-creates a "default" stage from `[training]` fields to cover the gap (steps 0-5). This lets you train initially with `[training]` only, then later add stages mid-training to test different data mixtures from a checkpoint.
6. **No data skip/repeat**: Same batches processed in same order
