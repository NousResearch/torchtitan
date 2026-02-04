#!/bin/bash
#SBATCH --job-name=bf16_dropc
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/bf16_drop_cache_%j.out
#SBATCH --error=logs/bf16_drop_cache_%j.err
#SBATCH --exclusive

# =============================================================================
# BF16 OPTIMIZER STATES + DROP PAGE CACHE TEST
# =============================================================================
# Testing hypothesis: dropping mmap'd model cache before training prevents OOM
#
# Memory equation WITHOUT cache drop:
#   510GB (optimizer bf16) + 1480GB (mmap cache) = 1990GB ≈ 2TB LIMIT → OOM
#
# Memory equation WITH cache drop:
#   510GB (optimizer bf16) + ~128GB (active shards only) = ~638GB << 2TB → OK?
#
# Key question: After dropping cache, does training only page in active shards?
# =============================================================================

set -x

echo "=========================================="
echo "BF16 Optimizer + Drop Cache Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_bf16_drop_cache"
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
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be13_bf16_drop_cache.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "Log dir: $LOG_DIR"
echo "Config: $CONFIG"

# =============================================================================
# PRE-TRAINING SYSTEM STATE
# =============================================================================
echo ""
echo "=== PRE-TRAINING SYSTEM STATE ($(date)) ==="
echo "--- Memory ---"
free -h
echo ""
echo "--- /proc/meminfo ---"
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|Writeback|AnonPages|Mapped|Shmem"
echo ""

# =============================================================================
# START COMPREHENSIVE BACKGROUND MONITORING
# =============================================================================
echo ""
echo "=== STARTING BACKGROUND MONITORS ==="

# 1. vmstat - 1 second intervals
echo "Starting vmstat (1s intervals)..."
vmstat -t 1 > "$LOG_DIR/vmstat_1s.log" 2>&1 &
VMSTAT_PID=$!
echo "  vmstat PID: $VMSTAT_PID"

# 2. Detailed memory info every 5 seconds (CSV format)
echo "Starting detailed memory monitor (5s intervals)..."
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

# 3. nvidia-smi - 1 second intervals
echo "Starting nvidia-smi (1s intervals)..."
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv -l 1 > "$LOG_DIR/nvidia_smi_1s.csv" 2>&1 &
NVIDIA_PID=$!
echo "  nvidia-smi PID: $NVIDIA_PID"

# 4. iostat - disk I/O
echo "Starting iostat (5s intervals)..."
iostat -x 5 > "$LOG_DIR/iostat.log" 2>&1 &
IOSTAT_PID=$!
echo "  iostat PID: $IOSTAT_PID"

# 5. Simple memory summary every 10 seconds
echo "Starting memory summary (10s intervals)..."
(
    while true; do
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_DIR/memory_summary.log"
        free -h >> "$LOG_DIR/memory_summary.log"
        echo "" >> "$LOG_DIR/memory_summary.log"
        sleep 10
    done
) &
MEMSUM_PID=$!
echo "  memory summary PID: $MEMSUM_PID"

# Save all PIDs
echo "$VMSTAT_PID $MEMINFO_PID $NVIDIA_PID $IOSTAT_PID $MEMSUM_PID" > "$LOG_DIR/monitor_pids.txt"
echo ""
echo "All monitors started. PIDs saved to $LOG_DIR/monitor_pids.txt"

sleep 3

# =============================================================================
# RUN TRAINING WITH PER-RANK LOGGING
# =============================================================================
echo ""
echo "=========================================="
echo "=== STARTING TRAINING ($(date)) ==="
echo "=========================================="
echo "Config: $CONFIG"
echo ""
echo "KEY SETTINGS:"
echo "  - optimizer.state_dtype = bfloat16"
echo "  - skip_optimizer_step = false (optimizer ENABLED)"
echo "  - local_batch_size = 4"
echo "  - seq_len = 32768"
echo "  - steps = 10"
echo "  - drop_page_cache_before_training = true  <-- NEW!"
echo ""
echo "EXPECTED BEHAVIOR:"
echo "  1. Model loads, cache fills to ~777GB"
echo "  2. Optimizer states pre-initialized in bf16 (~510GB AnonPages)"
echo "  3. >>> PAGE CACHE DROPPED <<< (cache should drop to near 0)"
echo "  4. Training starts, only active shards get paged in (~128GB?)"
echo "  5. Total: 510GB + ~128GB = ~638GB << 2TB"
echo "  6. All steps should complete without OOM!"
echo ""

# Run training with per-rank output files
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

# Stop monitors
echo "Stopping monitors..."
kill $(cat "$LOG_DIR/monitor_pids.txt") 2>/dev/null
sleep 2

# =============================================================================
# POST-TRAINING ANALYSIS
# =============================================================================
echo ""
echo "=========================================="
echo "=== ANALYSIS ==="
echo "=========================================="

echo ""
echo "--- Final Memory State ---"
free -h

echo ""
echo "--- Cache Drop Event (look for sudden cache drop) ---"
echo "Expected: Cache drops from ~777GB to near 0 after 'Page cache dropped'"
grep -E "drop|cache|Cache" "$LOG_DIR/per_rank/rank_0.log" 2>/dev/null | head -10

echo ""
echo "--- Memory Timeline (look for cache drop then gradual increase) ---"
echo "timestamp, Free_GB, Cache_GB, AnonPages_GB"
cat "$LOG_DIR/meminfo_detailed.csv" | awk -F',' 'NR>1 {printf "%s, %.0f, %.0f, %.0f\n", $1, $3/1024/1024, $6/1024/1024, $10/1024/1024}' | head -40

echo ""
echo "--- Peak Swap Activity ---"
echo "Columns: r=running, b=blocked, swpd=swap_used, si=swap_in, so=swap_out"
sort -t' ' -k8 -rn "$LOG_DIR/vmstat_1s.log" 2>/dev/null | grep -v "procs\|swpd\|^$" | head -10

echo ""
echo "--- GPU Memory (last 20 entries) ---"
tail -20 "$LOG_DIR/nvidia_smi_1s.csv"

echo ""
echo "--- Step Completion Check ---"
grep -E "step.*loss|STEP|completed" "$LOG_DIR/per_rank/rank_0.log" | head -20

echo ""
echo "=========================================="
echo "LOGS SAVED TO: $LOG_DIR/"
echo "=========================================="
ls -la "$LOG_DIR/"
echo ""
echo "Per-rank logs:"
ls -la "$LOG_DIR/per_rank/" | head -10

echo ""
echo "KEY QUESTIONS TO ANSWER:"
echo "  1. Did cache drop work? (Look for cache going from 777GB to ~0)"
echo "  2. How much did cache grow during training? (Should be << 1480GB)"
echo "  3. Did training complete all 10 steps?"
echo "  4. Any swap activity? (si/so should be 0)"
