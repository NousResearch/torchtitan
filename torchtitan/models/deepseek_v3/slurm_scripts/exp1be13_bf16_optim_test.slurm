#!/bin/bash
#SBATCH --job-name=exp1be13_bf16
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp1be13_bf16_%j.out
#SBATCH --error=logs/exp1be13_bf16_%j.err
#SBATCH --exclusive

set -x

echo "=========================================="
echo "exp1be13 with fast_init bf16 + optimizer step"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_exp1be13_bf16"
mkdir -p "$LOG_DIR"

HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8

cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be13_bf16_optim_test.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "Config: $CONFIG"

# Start vmstat monitoring
vmstat -t 1 > "$LOG_DIR/vmstat_1s.log" 2>&1 &
VMSTAT_PID=$!

# Start memory monitoring
(
    while true; do
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_DIR/memory.log"
        free -h >> "$LOG_DIR/memory.log"
        sleep 10
    done
) &
MEM_PID=$!

echo ""
echo "=== PRE-TRAINING MEMORY ==="
free -h

echo ""
echo "=== STARTING TRAINING ==="
echo "Key settings:"
echo "  - local_batch_size = 6"
echo "  - seq_len = 32768"
echo "  - skip_optimizer_step = false (ENABLED)"
echo "  - fast_init bf16 = enabled"
echo ""

timeout 3600 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))
        python -m torchtitan.train --job.config_file '"$CONFIG"'
    '

TRAIN_EXIT=$?

echo ""
echo "=== TRAINING FINISHED ==="
echo "Exit code: $TRAIN_EXIT"

# Stop monitors
kill $VMSTAT_PID $MEM_PID 2>/dev/null

echo ""
echo "=== POST-TRAINING MEMORY ==="
free -h

echo ""
echo "=== PEAK SWAP ACTIVITY ==="
sort -t' ' -k8 -rn "$LOG_DIR/vmstat_1s.log" 2>/dev/null | grep -v "procs\|swpd" | head -10

echo ""
echo "Logs saved to: $LOG_DIR/"
