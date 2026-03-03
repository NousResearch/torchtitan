#!/bin/bash
# Wrapper script to launch SLURM job with automatic health monitoring
#
# Usage:
#   ./run_with_monitor.sh online_multinode_vllm_test.slurm --auto-kill
#   ./run_with_monitor.sh test_full_rl.slurm

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SLURM_SCRIPT="$1"
AUTO_KILL_FLAG=""
CURRENT_DIR="$( pwd )"

if [ "$2" == "--auto-kill" ]; then
    AUTO_KILL_FLAG="--auto-kill"
fi

if [ -z "$SLURM_SCRIPT" ]; then
    echo "Usage: $0 <slurm_script> [--auto-kill]"
    exit 1
fi

echo "Submitting SLURM job: $SLURM_SCRIPT"
JOB_OUTPUT=$(sbatch --export=ALL,CONFIG_FILE=$CURRENT_DIR/torchtitan/grpo/configs/qwen25-7b-math.toml,MODEL_NAME=Qwen/Qwen2.5-7B,PYTHON_SCRIPT=/home/shared/atropos/environments/math_server_zero.py,WANDB_PROJECT=qwen7b_debug torchtitan/grpo/test/online_multinode_vllm_test.slurm)
JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job"
    exit 1
fi

echo "Job submitted: $JOB_ID"
echo "Starting health monitor..."
echo ""

python3 "$SCRIPT_DIR/monitor_training.py" --job-id "$JOB_ID" $AUTO_KILL_FLAG
