#!/bin/bash
#SBATCH --job-name=bf16_memlim
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1800G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bf16_mem_limit_%j.out
#SBATCH --error=logs/bf16_mem_limit_%j.err
#SBATCH --exclusive

# =============================================================================
# BF16 OPTIMIZER STATES + MEMORY LIMIT TEST (PER-JOB, NOT SYSTEM-WIDE)
# =============================================================================
# Using --mem=1800G to limit THIS JOB's memory via cgroup
# This forces kernel to reclaim page cache earlier for just this job
#
# Memory equation:
#   Available: 1800GB (limited by cgroup)
#   Optimizer: ~510GB (bf16)
#   Remaining for cache: ~1290GB
#
# If active working set < 1290GB, training should work!
# The cgroup limit forces cache eviction before OOM.
#
# THIS DOES NOT AFFECT OTHER USERS - only this job is limited.
# =============================================================================

set -x

echo "=========================================="
echo "BF16 Optimizer + Memory Limit Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "KEY: --mem=1800G limits THIS JOB to 1.8TB per node (vs 2TB available)"
echo "This creates a cgroup that forces earlier cache reclaim"
echo ""

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_bf16_mem_limit"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/per_rank"

HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be13_bf16_optim_states_batch4.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "Log dir: $LOG_DIR"
echo "Config: $CONFIG"

# Check cgroup memory limit
echo ""
echo "=== CGROUP MEMORY LIMIT CHECK ==="
CGROUP_LIMIT=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.limit_in_bytes 2>/dev/null || echo "not found")
echo "Cgroup memory limit: $CGROUP_LIMIT bytes"
if [ "$CGROUP_LIMIT" != "not found" ]; then
    echo "Cgroup memory limit: $((CGROUP_LIMIT / 1024 / 1024 / 1024)) GB"
fi

# =============================================================================
# PRE-TRAINING SYSTEM STATE
# =============================================================================
echo ""
echo "=== PRE-TRAINING SYSTEM STATE ($(date)) ==="
echo "--- Memory ---"
free -h
echo ""
echo "--- /proc/meminfo ---"
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|AnonPages"
echo ""

# =============================================================================
# START BACKGROUND MONITORING
# =============================================================================
echo ""
echo "=== STARTING BACKGROUND MONITORS ==="

vmstat -t 1 > "$LOG_DIR/vmstat_1s.log" 2>&1 &
VMSTAT_PID=$!
echo "  vmstat PID: $VMSTAT_PID"

(
    echo "timestamp,MemTotal_KB,MemFree_KB,MemAvailable_KB,Buffers_KB,Cached_KB,SwapTotal_KB,SwapFree_KB,SwapUsed_KB,AnonPages_KB,Dirty_KB" > "$LOG_DIR/meminfo_detailed.csv"
    while true; do
        TS=$(date '+%Y-%m-%d %H:%M:%S')
        MEMTOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        MEMFREE=$(grep "MemFree:" /proc/meminfo | awk '{print $2}')
        MEMAVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        BUFFERS=$(grep "Buffers:" /proc/meminfo | awk '{print $2}')
        CACHED=$(grep "^Cached:" /proc/meminfo | awk '{print $2}')
        SWAPTOTAL=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
        SWAPFREE=$(grep SwapFree /proc/meminfo | awk '{print $2}')
        SWAPUSED=$((SWAPTOTAL - SWAPFREE))
        ANONPAGES=$(grep AnonPages /proc/meminfo | awk '{print $2}')
        DIRTY=$(grep "Dirty:" /proc/meminfo | awk '{print $2}')
        echo "$TS,$MEMTOTAL,$MEMFREE,$MEMAVAIL,$BUFFERS,$CACHED,$SWAPTOTAL,$SWAPFREE,$SWAPUSED,$ANONPAGES,$DIRTY" >> "$LOG_DIR/meminfo_detailed.csv"
        sleep 5
    done
) &
MEMINFO_PID=$!
echo "  meminfo PID: $MEMINFO_PID"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv -l 1 > "$LOG_DIR/nvidia_smi_1s.csv" 2>&1 &
NVIDIA_PID=$!
echo "  nvidia-smi PID: $NVIDIA_PID"

iostat -x 5 > "$LOG_DIR/iostat.log" 2>&1 &
IOSTAT_PID=$!
echo "  iostat PID: $IOSTAT_PID"

echo "$VMSTAT_PID $MEMINFO_PID $NVIDIA_PID $IOSTAT_PID" > "$LOG_DIR/monitor_pids.txt"
echo "All monitors started."

sleep 3

# =============================================================================
# RUN TRAINING
# =============================================================================
echo ""
echo "=========================================="
echo "=== STARTING TRAINING ($(date)) ==="
echo "=========================================="
echo "Config: $CONFIG"
echo ""
echo "KEY SETTINGS:"
echo "  - --mem=1800G (SLURM memory limit per node)"
echo "  - optimizer.state_dtype = bfloat16"
echo "  - skip_optimizer_step = false (optimizer ENABLED)"
echo "  - local_batch_size = 4"
echo "  - seq_len = 32768"
echo "  - steps = 10"
echo ""
echo "EXPECTED BEHAVIOR:"
echo "  1. Cgroup limits job to 1.8TB"
echo "  2. Optimizer states use ~510GB"
echo "  3. Remaining ~1290GB for cache"
echo "  4. Kernel reclaims cache BEFORE hitting limit"
echo "  5. No OOM, training completes!"
echo ""

timeout 3600 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))
        LOG_FILE="'"$LOG_DIR"'/per_rank/rank_${RANK}.log"
        echo "=== RANK $RANK starting at $(date) ===" > "$LOG_FILE"
        echo "Node: $(hostname)" >> "$LOG_FILE"
        echo "LOCAL_RANK: $LOCAL_RANK" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        python -m torchtitan.train --job.config_file '"$CONFIG"' 2>&1 | tee -a "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        echo "=== RANK $RANK finished at $(date) ===" >> "$LOG_FILE"
    '

TRAIN_EXIT=$?

# =============================================================================
# POST-TRAINING
# =============================================================================
echo ""
echo "=========================================="
echo "=== TRAINING FINISHED ($(date)) ==="
echo "=========================================="
echo "Exit code: $TRAIN_EXIT"
if [ $TRAIN_EXIT -eq 124 ]; then
    echo "STATUS: TIMEOUT"
elif [ $TRAIN_EXIT -eq 0 ]; then
    echo "STATUS: SUCCESS"
else
    echo "STATUS: FAILED"
fi
echo ""

kill $(cat "$LOG_DIR/monitor_pids.txt") 2>/dev/null
sleep 2

# =============================================================================
# ANALYSIS
# =============================================================================
echo ""
echo "=========================================="
echo "=== ANALYSIS ==="
echo "=========================================="

echo ""
echo "--- Final Memory State ---"
free -h

echo ""
echo "--- Memory Timeline (Free, Cache, AnonPages in GB) ---"
cat "$LOG_DIR/meminfo_detailed.csv" | awk -F',' 'NR>1 {printf "%s, Free=%.0f, Cache=%.0f, Anon=%.0f\n", $1, $3/1024/1024, $6/1024/1024, $10/1024/1024}' | head -50

echo ""
echo "--- Peak Cache Usage ---"
cat "$LOG_DIR/meminfo_detailed.csv" | awk -F',' 'NR>1 {print $6}' | sort -rn | head -1 | awk '{printf "Peak cache: %.0f GB\n", $1/1024/1024}'

echo ""
echo "--- Swap Activity (should be 0) ---"
grep -v "procs\|swpd" "$LOG_DIR/vmstat_1s.log" | awk '$8 > 0 || $9 > 0 {print}' | head -10 || echo "No swap activity detected!"

echo ""
echo "--- Step Completion ---"
grep -E "step.*loss|STEP|completed" "$LOG_DIR/per_rank/rank_0.log" | head -20

echo ""
echo "=========================================="
echo "LOGS: $LOG_DIR/"
echo "=========================================="
ls -la "$LOG_DIR/"
