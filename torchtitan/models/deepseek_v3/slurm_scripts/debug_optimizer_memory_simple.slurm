#!/bin/bash
#SBATCH --job-name=debug_optim_mem
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_optim_mem_%j.out
#SBATCH --error=logs/debug_optim_mem_%j.err
#SBATCH --exclusive

# =============================================================================
# DEBUG: Verify CPU Memory Exhaustion Hypothesis (Simple Shell-Based Monitoring)
# =============================================================================

set -x

echo "=========================================="
echo "DEBUG: Optimizer Memory Test (Simple)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_debug"
mkdir -p "$LOG_DIR"

HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# Get head node IP
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8

cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/kimi_1t_debug_optimizer_memory.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "World size: $WORLD_SIZE"

# =============================================================================
# PRE-TRAINING MEMORY
# =============================================================================
echo ""
echo "=== PRE-TRAINING MEMORY ($(date)) ==="
free -h
echo ""

# =============================================================================
# START BACKGROUND MONITORING (simple shell-based)
# =============================================================================
echo "Starting vmstat monitoring..."
vmstat -t 5 > "$LOG_DIR/vmstat.log" 2>&1 &
VMSTAT_PID=$!

echo "Starting nvidia-smi monitoring..."
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv -l 5 > "$LOG_DIR/nvidia_smi.csv" 2>&1 &
NVIDIA_PID=$!

# Memory logging every 10 seconds
(
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -g | grep Mem | awk '{print "RAM: "$3"/"$2" GB ("int($3/$2*100)"%)"}')" >> "$LOG_DIR/memory.log"
        echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -g | grep Swap | awk '{print "Swap: "$3"/"$2" GB"}')" >> "$LOG_DIR/memory.log"
        sleep 10
    done
) &
MEM_LOG_PID=$!

echo "Monitors started: vmstat=$VMSTAT_PID, nvidia=$NVIDIA_PID, memlog=$MEM_LOG_PID"

# =============================================================================
# RUN TRAINING
# =============================================================================
echo ""
echo "=== STARTING TRAINING ($(date)) ==="
echo "Config: $CONFIG"
echo "Watch for: [OPTIM DEBUG] lines in output for memory usage"
echo ""

srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))
        python -m torchtitan.train --job.config_file '"$CONFIG"'
    '

TRAIN_EXIT=$?

# =============================================================================
# POST-TRAINING
# =============================================================================
echo ""
echo "=== TRAINING FINISHED (exit code: $TRAIN_EXIT) ($(date)) ==="
echo ""
echo "=== POST-TRAINING MEMORY ==="
free -h
echo ""

# Stop monitors
kill $VMSTAT_PID $NVIDIA_PID $MEM_LOG_PID 2>/dev/null

# =============================================================================
# ANALYSIS
# =============================================================================
echo "=== MEMORY LOG ==="
cat "$LOG_DIR/memory.log"
echo ""

echo "=== VMSTAT SUMMARY (last 20 lines) ==="
tail -20 "$LOG_DIR/vmstat.log"
echo ""

echo "=== GPU MEMORY (last 20 lines) ==="
tail -20 "$LOG_DIR/nvidia_smi.csv"
echo ""

echo "=========================================="
echo "Logs saved to: $LOG_DIR/"
echo "=========================================="
