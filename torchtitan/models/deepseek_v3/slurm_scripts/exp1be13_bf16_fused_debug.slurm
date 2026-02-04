#!/bin/bash
#SBATCH --job-name=bf16_fused_debug
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/bf16_fused_debug_%j.out
#SBATCH --error=logs/bf16_fused_debug_%j.err
#SBATCH --exclusive

# =============================================================================
# BF16 OPTIMIZER STATES + FUSED ADAM DEBUG TEST
# =============================================================================
# Testing if fused Adam works when optimizer states match param dtype.
# The preinit_optimizer_states_bf16() now:
#   1. Logs first 5 param dtypes, devices, shapes
#   2. Logs first 3 state allocations showing param vs state dtype
#   3. Allocates states with dtype=p.dtype (not hardcoded bf16)
# =============================================================================

set -x

echo "=========================================="
echo "BF16 Optimizer + Fused Adam Debug Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_bf16_fused_debug"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/per_rank"

HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8

# Standard debug settings
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN

# EXPLICIT LONG TIMEOUTS via environment variables
export NCCL_TIMEOUT=1800000        # 30 minutes in milliseconds
export TORCH_NCCL_BLOCKING_WAIT=1  # Use blocking wait
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3  # Skip cleanup on error for debugging

echo ""
echo "TIMEOUT SETTINGS:"
echo "  NCCL_TIMEOUT=$NCCL_TIMEOUT ms (30 min)"
echo "  Config: init_timeout_seconds=1800, train_timeout_seconds=1800"
echo ""
echo "DEBUG FOCUS:"
echo "  - Check param dtype in optimizer state pre-initialization logs"
echo "  - Verify states are allocated with same dtype as params"
echo "  - Test if fused Adam works with matching dtypes"
echo ""

cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be13_bf16_fused_debug.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "Log dir: $LOG_DIR"
echo "Config: $CONFIG"

# =============================================================================
# PRE-TRAINING STATE
# =============================================================================
echo ""
echo "=== PRE-TRAINING SYSTEM STATE ($(date)) ==="
free -h
echo ""

# =============================================================================
# BACKGROUND MONITORS
# =============================================================================
vmstat -t 1 > "$LOG_DIR/vmstat_1s.log" 2>&1 &
VMSTAT_PID=$!

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
        sleep 10
    done
) &
MEMINFO_PID=$!

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv -l 5 > "$LOG_DIR/nvidia_smi_5s.csv" 2>&1 &
NVIDIA_PID=$!

echo "$VMSTAT_PID $MEMINFO_PID $NVIDIA_PID" > "$LOG_DIR/monitor_pids.txt"
echo "Monitors started."

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
echo "  - init_timeout_seconds = 1800 (30 min)"
echo "  - train_timeout_seconds = 1800 (30 min)"
echo "  - NCCL_TIMEOUT = 1800000 ms (30 min)"
echo "  - steps = 3"
echo "  - optimizer.state_dtype = bfloat16"
echo "  - optimizer.implementation = fused (default)"
echo ""
echo "EXPECTED: Detailed logging of param dtype will show if params are bf16."
echo "          If states match param dtype, fused Adam should work."
echo ""

timeout 7200 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
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

kill $(cat "$LOG_DIR/monitor_pids.txt") 2>/dev/null
sleep 2

# =============================================================================
# ANALYSIS
# =============================================================================
echo ""
echo "=== ANALYSIS ==="

echo ""
echo "--- PARAM DTYPE DEBUG INFO (look for 'param dtype' and 'State init sample') ---"
grep -E "param\[|param dtype|State init sample|Pre-initialized" "$LOG_DIR/per_rank/rank_0.log" | head -30

echo ""
echo "--- Step Completion Check ---"
grep -E "step.*loss|STEP|Duration|completed" "$LOG_DIR/per_rank/rank_0.log" | head -20

echo ""
echo "--- Memory Timeline (last 30 entries) ---"
tail -30 "$LOG_DIR/meminfo_detailed.csv" | awk -F',' 'NR>1 {printf "%s Free=%.0fGB Cache=%.0fGB Anon=%.0fGB\n", $1, $3/1024/1024, $6/1024/1024, $10/1024/1024}'

echo ""
echo "--- Any Errors? ---"
grep -iE "error|exception|fail|RuntimeError|dtype" "$LOG_DIR/per_rank/rank_0.log" | tail -20

echo ""
echo "=========================================="
echo "LOGS: $LOG_DIR/"
echo "=========================================="
ls -la "$LOG_DIR/"
