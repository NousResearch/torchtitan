#!/bin/bash
#SBATCH --job-name=cmp_mdl_loss
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/compile_model_loss_%j.out
#SBATCH --error=logs/compile_model_loss_%j.err
#SBATCH --exclusive

set -x

echo "=========================================="
echo "exp1be14 Full BF16 + Compile Model + Loss"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="
echo "Config: 8n, EP64, CP8, LBS6, ctx32k"
echo "Compile: model + loss"
echo ""

mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_compile_model_loss"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/per_rank"

HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3

cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/exp1be14_full_bf16_compile_model_loss.toml"

echo "Config: $CONFIG"
echo "compile.enable = true"
echo "compile.components = [\"model\", \"loss\"]"

# Memory monitor
(
    echo "timestamp,MemFree_KB,MemAvailable_KB,Cached_KB,AnonPages_KB" > "$LOG_DIR/meminfo.csv"
    while true; do
        TS=$(date '+%Y-%m-%d %H:%M:%S')
        MEMFREE=$(grep "MemFree:" /proc/meminfo | awk '{print $2}')
        MEMAVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        CACHED=$(grep "^Cached:" /proc/meminfo | awk '{print $2}')
        ANONPAGES=$(grep AnonPages /proc/meminfo | awk '{print $2}')
        echo "$TS,$MEMFREE,$MEMAVAIL,$CACHED,$ANONPAGES" >> "$LOG_DIR/meminfo.csv"
        sleep 10
    done
) &
MEMINFO_PID=$!

timeout 7200 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))
        LOG_FILE="'"$LOG_DIR"'/per_rank/rank_${RANK}.log"
        python -m torchtitan.train --job.config_file '"$CONFIG"' 2>&1 | tee "$LOG_FILE"
    '

TRAIN_EXIT=$?
kill $MEMINFO_PID 2>/dev/null

echo ""
echo "=== RESULTS ==="
echo "Exit code: $TRAIN_EXIT"
grep -E "step:.*loss" "$LOG_DIR/per_rank/rank_0.log" | head -10
echo ""
echo "Memory timeline:"
tail -10 "$LOG_DIR/meminfo.csv" | awk -F',' '{printf "%s Free=%.0fGB Cache=%.0fGB\n", $1, $2/1024/1024, $4/1024/1024}'
