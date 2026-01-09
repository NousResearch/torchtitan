# Online Evaluation with lm-evaluation-harness

This document describes how to run automatic evaluations during training using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Overview

Online evaluation allows you to track model performance on standard benchmarks throughout training, providing insights into:
- Training progress and convergence
- Potential overfitting or underfitting
- Optimal checkpoint selection

## Features

- **Three execution modes**: inline (blocking), subprocess (background), SLURM (async)
- **Full reproducibility**: Seed control for all random number generators
- **Automatic checkpoint evaluation**: Runs after checkpoint saves at specified intervals
- **Generated scripts**: Standalone Python scripts for reproducibility and debugging

## Configuration

Add the `[lm_eval]` section to your training config:

```toml
[lm_eval]
enable = true
tasks = "hellaswag,arc_easy"
eval_interval = 500          # Run eval every N steps
num_fewshot = 0
limit = 100                  # Samples per task (None = full eval)
batch_size = 4
mode = "slurm"               # "inline", "subprocess", or "slurm"

[lm_eval.slurm]
partition = "batch"
time = "01:00:00"
gpus_per_node = 1
```

## Execution Modes

### 1. Inline Mode (`mode = "inline"`)

Runs evaluation in the same process as training. Simple but **blocks training** during evaluation.

**Best for**: Quick tests, small models, or when GPU resources are limited.

```toml
[lm_eval]
enable = true
mode = "inline"
limit = 50  # Use small limit to reduce blocking time
```

### 2. Subprocess Mode (`mode = "subprocess"`)

Runs evaluation in a background subprocess on the same node. **Non-blocking** but shares resources with training.

**Best for**: Development and testing on single nodes.

```toml
[lm_eval]
enable = true
mode = "subprocess"
```

### 3. SLURM Mode (`mode = "slurm"`)

Submits evaluation as a separate SLURM job. **Fully async** with dedicated resources.

**Best for**: Production training on clusters.

```toml
[lm_eval]
enable = true
mode = "slurm"
job_name_prefix = "my_experiment_eval"

[lm_eval.slurm]
partition = "batch"
time = "02:00:00"
gpus_per_node = 1
cpus_per_task = 16
hf_cache = "/home/shared/huggingface-cache"
```

## Configuration Reference

### LMEvalConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | false | Enable automatic evaluation |
| `eval_interval` | int | 500 | Run eval every N steps (must align with checkpoint.interval) |
| `tasks` | str | "hellaswag,arc_easy" | Comma-separated lm-eval tasks |
| `num_fewshot` | int | 0 | Number of few-shot examples |
| `limit` | int\|None | None | Samples per task (None = full) |
| `batch_size` | int | 4 | Evaluation batch size |
| `max_seq_len` | int | 2048 | Maximum sequence length |
| `mode` | str | "inline" | Execution mode |
| `seed` | int | 42 | Base random seed |
| `output_dir` | str | "eval_results" | Results directory (relative to dump_folder) |
| `log_samples` | bool | true | Log individual predictions |
| `job_name_prefix` | str | "lm_eval" | SLURM job name prefix |

### LMEvalSlurmConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `partition` | str | "batch" | SLURM partition |
| `gpus_per_node` | int | 1 | GPUs for eval job |
| `cpus_per_task` | int | 16 | CPUs per task |
| `time` | str | "02:00:00" | Time limit |
| `qos` | str\|None | None | Quality of service |
| `account` | str\|None | None | SLURM account |
| `hf_cache` | str | "/home/shared/huggingface-cache" | HuggingFace cache path |
| `venv_path` | str\|None | None | Python venv path |
| `conda_env` | str\|None | None | Conda environment name |

## Output Structure

After training with online eval, your dump folder will contain:

```
dump_folder/
├── checkpoint/
│   ├── step-500/
│   └── step-1000/
├── eval_results/
│   ├── step_500/
│   │   ├── eval_config.json    # Evaluation configuration
│   │   ├── results.json        # lm-eval results
│   │   └── run_eval.py         # Standalone eval script
│   └── step_1000/
│       └── ...
├── eval_slurm_scripts/         # (SLURM mode only)
│   ├── eval_step_500.sh
│   └── eval_step_1000.sh
└── eval_slurm_logs/            # (SLURM mode only)
    ├── lm_eval_step_500_12345.out
    └── lm_eval_step_500_12345.err
```

## Re-running Evaluations

Each evaluation generates a standalone script that can be re-run:

```bash
# Re-run a specific evaluation
python /path/to/dump_folder/eval_results/step_500/run_eval.py

# Or resubmit the SLURM job
sbatch /path/to/dump_folder/eval_slurm_scripts/eval_step_500.sh
```

## Quick Test

Use the provided test config to verify online eval works:

```bash
# Set paths (adjust to your environment)
export TORCHTITAN_PATH="/home/phuc/workspace/moe/online_evals/torchtitan"
export LM_EVAL_PATH="/home/phuc/workspace/moe/online_evals/lm-evaluation-harness"
export PYTHON_ENV="/home/phuc/workspace/moe/online_evals/env/bin"

# Run test: 20 training steps with eval at steps 5, 10, 15, 20
PYTHONPATH="${TORCHTITAN_PATH}:${LM_EVAL_PATH}:$PYTHONPATH" \
${PYTHON_ENV}/torchrun --nproc_per_node=1 --standalone \
  -m torchtitan.train \
  --job.config-file ${TORCHTITAN_PATH}/torchtitan/models/llama3/train_configs/online_eval_test.toml
```

The test config (`torchtitan/models/llama3/train_configs/online_eval_test.toml`) runs:
- 20 training steps on the c4_test dataset with llama3 debugmodel
- Checkpoints saved every 5 steps
- Evaluation (hellaswag, 20 samples) after each checkpoint
- SLURM mode (async) for non-blocking evaluation

**Expected output:**
```
[titan] Training starts at step 1
[titan] step:  5  loss: ...
[titan] Saving the checkpoint...
[titan] Starting evaluation at step 5
[titan] Evaluation results at step 5:
[titan]   hellaswag: acc,none=0.XXXX, acc_norm,none=0.XXXX
...
```

## Troubleshooting

### Eval interval doesn't match checkpoint interval

The `eval_interval` must be a multiple of `checkpoint.interval`. Evaluation runs after checkpoints are saved.

### SLURM job fails with "python not found"

Set `venv_path` in SLURM config to your Python environment:

```toml
[lm_eval.slurm]
venv_path = "/path/to/your/venv"
```

### Out of memory during inline eval

- Reduce `batch_size`
- Use `mode = "slurm"` to run on separate resources
- Set `limit` to reduce number of samples

### Evaluation results not appearing

Check the SLURM logs in `eval_slurm_logs/` for errors. Common issues:
- Missing HuggingFace cache permissions
- Incompatible model/tokenizer paths
