#!/bin/bash
#SBATCH --job-name=bf16_cmp_nm
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/bf16_compile_nomem_%j.out
#SBATCH --error=logs/bf16_compile_nomem_%j.err
#SBATCH --exclusive

# BF16 + compile loss + NO mem limit (--mem=0)
set -x

echo "=========================================="
echo "BF16 + Compile + NO Mem Limit"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_bf16_compile_nomem"
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
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be13_bf16_compile.toml"

echo "Head node: $HEAD_NODE ($MASTER_ADDR)"
echo "Log dir: $LOG_DIR"
echo "Config: $CONFIG"
echo ""
echo "KEY SETTINGS:"
echo "  --mem=0 (NO memory limit)"
echo "  compile.enable = true"
echo "  compile.components = [loss]"
echo "  comm.init_timeout_seconds = 900"
echo ""

echo "=== PRE-TRAINING STATE ==="
free -h
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Cached|AnonPages"
echo ""

# Monitors
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
        sleep 5
    done
) &
MEMINFO_PID=$!

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv -l 1 > "$LOG_DIR/nvidia_smi_1s.csv" 2>&1 &
NVIDIA_PID=$!

echo "$VMSTAT_PID $MEMINFO_PID $NVIDIA_PID" > "$LOG_DIR/monitor_pids.txt"
sleep 2

echo "=== STARTING TRAINING ($(date)) ==="

timeout 3600 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))
        LOG_FILE="'"$LOG_DIR"'/per_rank/rank_${RANK}.log"
        echo "=== RANK $RANK starting at $(date) ===" > "$LOG_FILE"
        echo "Node: $(hostname)" >> "$LOG_FILE"
        python -m torchtitan.train --job.config_file '"$CONFIG"' 2>&1 | tee -a "$LOG_FILE"
        echo "=== RANK $RANK finished at $(date) ===" >> "$LOG_FILE"
    '

TRAIN_EXIT=$?

echo ""
echo "=== TRAINING FINISHED ($(date)) ==="
echo "Exit code: $TRAIN_EXIT"
[ $TRAIN_EXIT -eq 0 ] && echo "STATUS: SUCCESS" || echo "STATUS: FAILED"

kill $(cat "$LOG_DIR/monitor_pids.txt") 2>/dev/null

echo ""
echo "=== ANALYSIS ==="
free -h
echo ""
echo "Memory timeline (last 30):"
cat "$LOG_DIR/meminfo_detailed.csv" | awk -F',' 'NR>1 {printf "%s Free=%.0fGB Cache=%.0fGB Anon=%.0fGB\n", $1, $3/1024/1024, $6/1024/1024, $10/1024/1024}' | tail -30
echo ""
echo "Step completion:"
grep -E "step.*loss|STEP|Duration" "$LOG_DIR/per_rank/rank_0.log" | head -20
echo ""
echo "LOGS: $LOG_DIR/"
