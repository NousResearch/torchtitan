# Multi-Stage Data Training

Multi-stage training allows switching between different data mixtures at specified training steps, similar to approaches used in Qwen3, DeepSeek-V3, and Llama 3.

## Quick Start

Add `[[training.data_stages]]` sections to your TOML config. Each stage is self-contained and defines all its data configuration:

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

Each `[[training.data_stages]]` section should define all data-related fields explicitly:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Stage identifier for logging |
| `start_step` | int | Yes | Step when stage begins (inclusive) |
| `end_step` | int | No | Step when stage ends (exclusive). Omit for final stage |
| `dataset` | string | Yes* | Dataset name (for huggingface type) |
| `dataset_path` | string | No | Path to dataset |
| `dataset_type` | string | Yes* | `"huggingface"`, `"nanoset"`, `"preprocessed"`, `"packed_memmap"` |
| `dataset_folders` | list | Yes* | Folders for nanoset datasets |
| `dataset_weights` | list | No | Weights for blending datasets |
| `dataset_random_seed` | int | No | Random seed for this stage |
| `seq_len` | int | Yes* | Sequence length |

*Required for each stage to be self-contained. Falls back to `[training]` if not set.

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

## Backward Compatibility

- If no `[[training.data_stages]]` sections are defined, training runs as single-stage (existing behavior)
- All existing configs work without modification

## Testing

### Automated Test

Run the test script to verify stage transitions, checkpoint save/resume, and exact reproducibility:

```bash
./scripts/test_data_stages.sh
```

Expected output:
```
[Test 1] Full run: steps 1-15 with 3 stage transitions
  ✓ Transition at step 5: stage_1_general -> stage_2_reasoning
  ✓ Transition at step 10: stage_2_reasoning -> stage_3_final

[Test 2] Resume run: from checkpoint at step 7
  ✓ Stage correctly restored to stage_2_reasoning
  ✓ Dataloader position restored
  ✓ Training resumed at correct step (8)

[Test 3] Reproducibility: comparing losses between full and resumed runs
Step  | Full Run | Resume   | Match
------|----------|----------|------
8     | 4.7074   | 4.7074   | ✓
...
15    | 3.7097   | 3.7097   | ✓

SUCCESS: All tests passed!
```

### Manual Testing

A test config is provided at `torchtitan/models/llama3/train_configs/data_stages_test.toml`.

```bash
# Full run
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file torchtitan/models/llama3/train_configs/data_stages_test.toml

# Resume from step 7
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file torchtitan/models/llama3/train_configs/data_stages_test.toml \
    --checkpoint.load_step 7
```

### What the Test Verifies

1. **Stage transitions**: Dataloader rebuilds at step 5 and 10
2. **Checkpoint saves**: Stage index + exact dataloader position (sample count)
3. **Resume restores**: Exact state - losses match between full run and resumed run
4. **No data skip/repeat**: Same batches processed in same order

### Verified Test Results

Results from running `./scripts/test_data_stages.sh`, comparing a full run (steps 1-15) vs resumed run (checkpoint at step 7, resume steps 8-15):

| Step | Full Run Loss | Resume Loss | Full Grad Norm | Resume Grad Norm |
|------|---------------|-------------|----------------|------------------|
| 8    | 4.7074        | 4.7074      | 1.6950         | 1.6950           |
| 9    | 4.0312        | 4.0312      | 1.9540         | 1.9540           |
| 10   | 4.0549        | 4.0549      | 1.5458         | 1.5458           |
| 11   | 3.8140        | 3.8140      | 1.5492         | 1.5492           |
| 12   | 3.8702        | 3.8702      | 1.4857         | 1.4857           |
| 13   | 4.2307        | 4.2307      | 1.3210         | 1.3210           |
| 14   | 3.6352        | 3.6352      | 1.4496         | 1.4496           |
| 15   | 3.7097        | 3.7097      | 1.3688         | 1.3688           |

All values match exactly, proving the dataloader position within the stage is correctly saved and restored.
