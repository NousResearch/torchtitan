#!/bin/bash
#SBATCH --job-name=prof_deepep
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --time=01:00:00
#SBATCH --output=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/prof_deepep_%j.out
#SBATCH --error=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/prof_deepep_%j.err

# Comprehensive profiling of DeepEP LBS=1 (1,210 TPS config)
# Collects: PyTorch profiler traces (memory+flops+stack+modules),
#           memory snapshots, nsys timeline, nvidia-smi dmon, GPU health

TORCHTITAN_DIR="/home/phuc/workspace/moe/small_prs/pr011_kimi_1t_profiling/torchtitan"
VENV_PATH="${TORCHTITAN_DIR}/.venv"
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/profiling_deepep_lbs1_8k_cp1.toml"
OUTPUT_DIR="${TORCHTITAN_DIR}/outputs/profiling_deepep_lbs1_1210tps"

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
NNODES=${#NODES[@]}

echo "============================================================"
echo "COMPREHENSIVE PROFILING: DeepEP LBS=1 (1,210 TPS baseline)"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $NNODES (${NODES[*]})"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "Config:       $CONFIG_FILE"
echo "Output:       $OUTPUT_DIR"
echo "Start time:   $(date)"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}/gpu_metrics"
mkdir -p "${OUTPUT_DIR}/nsys"
mkdir -p "${OUTPUT_DIR}/gpu_health"

# Phase 0: Pre-training GPU health
echo "[Phase 0] GPU health snapshot..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv > ${OUTPUT_DIR}/gpu_health/node${i}_${node}_pre.csv 2>&1" &
done
wait
echo "[Phase 0] Done."

# Phase 1: Background GPU monitoring
echo "[Phase 1] Starting GPU monitoring..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "nohup nvidia-smi dmon -s pucvmet -d 1 -f ${OUTPUT_DIR}/gpu_metrics/node${i}_${node}_dmon.csv > /dev/null 2>&1 & echo \$! > /tmp/dmon_pid_${SLURM_JOB_ID}"
done
echo "[Phase 1] Done."

# Phase 2: Training with profiling
echo "[Phase 2] Launching training..."

ENV_VARS="export LOGLEVEL=INFO && export FI_PROVIDER=efa && export NCCL_DEBUG=WARN && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib/:\$LD_LIBRARY_PATH && export NCCL_SOCKET_IFNAME=bond0 && export NCCL_BUFFSIZE=2097152 && export TORCH_DIST_INIT_BARRIER=1 && export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 && export HF_HOME=/tmp/hf_cache && export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && export NCCL_NVTX_ENABLE=1 && export TORCH_NCCL_TRACE_BUFFER_SIZE=1000 && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"

TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} -m torchtitan.train --job.config_file ${CONFIG_FILE}"

TRAIN_PIDS=()

# Master: nsys + torchrun
node="${NODES[0]}"
echo "  Master ${node}: nsys + torchrun"
ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && ${ENV_VARS} && nsys profile --trace=cuda,nvtx,cublas,cudnn,osrt --cuda-memory-usage=true --stats=true --force-overwrite=true --output=${OUTPUT_DIR}/nsys/master_${node} --sample=none ${TORCHRUN_CMD} 2>&1 | tee ${OUTPUT_DIR}/nsys/master_stdout.log" &
TRAIN_PIDS+=($!)

# Workers
for i in $(seq 1 $((NNODES - 1))); do
    node="${NODES[$i]}"
    echo "  Worker ${node}"
    ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && ${ENV_VARS} && ${TORCHRUN_CMD} 2>&1 | tee ${OUTPUT_DIR}/worker_${node}.log" &
    TRAIN_PIDS+=($!)
done

echo "[Phase 2] Waiting for completion..."
TRAIN_EXIT=0
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid" || TRAIN_EXIT=$?; done
echo "[Phase 2] Done (exit: $TRAIN_EXIT)"

# Phase 3: Stop monitoring + post health
echo "[Phase 3] Cleanup..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "if [ -f /tmp/dmon_pid_${SLURM_JOB_ID} ]; then kill \$(cat /tmp/dmon_pid_${SLURM_JOB_ID}) 2>/dev/null; rm -f /tmp/dmon_pid_${SLURM_JOB_ID}; fi; nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total --format=csv > ${OUTPUT_DIR}/gpu_health/node${i}_${node}_post.csv 2>&1" &
done
wait

# Phase 4: nsys stats
echo "[Phase 4] Generating nsys statistics..."
NSYS_REP="${OUTPUT_DIR}/nsys/master_${NODES[0]}.nsys-rep"
if [ -f "$NSYS_REP" ]; then
    nsys stats --report cuda_gpu_kern_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/cuda_kernel_summary.txt" 2>&1 || true
    nsys stats --report cuda_api_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/cuda_api_summary.txt" 2>&1 || true
    nsys stats --report nvtx_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/nvtx_summary.txt" 2>&1 || true
    echo "  Done."
else
    echo "  WARNING: nsys report not found"
    ls -la "${OUTPUT_DIR}/nsys/" 2>/dev/null
fi

echo ""
echo "============================================================"
echo "PROFILING COMPLETE — $(date)"
echo "============================================================"
echo "Traces: $(find ${OUTPUT_DIR} -name '*trace.json' | wc -l) files"
echo "Memory: $(find ${OUTPUT_DIR} -name '*memory_snapshot*' | wc -l) files"
echo "nsys:   $(ls ${OUTPUT_DIR}/nsys/*.nsys-rep 2>/dev/null | wc -l) reports"
