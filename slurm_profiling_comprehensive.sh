#!/bin/bash
#SBATCH --job-name=kimi_profile_lbs2
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --time=01:00:00
#SBATCH --output=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/profile_lbs2_%j.out
#SBATCH --error=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/profile_lbs2_%j.err

# =============================================================================
# Comprehensive Profiling: Kimi K2, LBS=2, seq=8k, CP=1, EP=64 (726 TPS)
# Collects: PyTorch profiler traces (memory + flops + stack + modules),
#           memory snapshots, nsys timeline, nvidia-smi dmon, GPU health
# =============================================================================

TORCHTITAN_DIR="/home/phuc/workspace/moe/small_prs/pr011_kimi_1t_profiling/torchtitan"
VENV_PATH="${TORCHTITAN_DIR}/.venv"
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/profiling_lbs2_8k_cp1_comprehensive.toml"
OUTPUT_DIR="${TORCHTITAN_DIR}/outputs/profiling_lbs2_726tps"

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
NNODES=${#NODES[@]}

echo "============================================================"
echo "COMPREHENSIVE PROFILING JOB"
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

# =============================================================================
# Phase 0: Pre-training GPU health snapshot
# =============================================================================
echo ""
echo "[Phase 0] Collecting pre-training GPU health..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv > ${OUTPUT_DIR}/gpu_health/node${i}_${node}_pre_training.csv 2>&1; nvidia-smi topo -m > ${OUTPUT_DIR}/gpu_health/node${i}_${node}_topo.txt 2>&1" &
done
wait
echo "[Phase 0] Done."

# =============================================================================
# Phase 1: Start background GPU monitoring on ALL nodes (non-blocking)
# =============================================================================
echo ""
echo "[Phase 1] Starting GPU monitoring on all nodes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    # nohup + redirect + & ensures SSH returns immediately
    ssh "$node" "nohup nvidia-smi dmon -s pucvmet -d 1 -f ${OUTPUT_DIR}/gpu_metrics/node${i}_${node}_dmon.csv > /dev/null 2>&1 & echo \$! > /tmp/dmon_pid_${SLURM_JOB_ID}"
done
echo "[Phase 1] Done. GPU monitoring running on ${NNODES} nodes."

# =============================================================================
# Phase 2: Launch training with profiling
# =============================================================================
echo ""
echo "[Phase 2] Launching profiling training..."
echo "  Master (${NODES[0]}): nsys + torchrun + PyTorch profiler + memory snapshot"
echo "  Workers (${NNODES}-1 nodes): torchrun + PyTorch profiler + memory snapshot"
echo ""

TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} -m torchtitan.train --job.config_file ${CONFIG_FILE}"

ENV_VARS="export LOGLEVEL=INFO && export FI_PROVIDER=efa && export NCCL_DEBUG=WARN && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib/:\$LD_LIBRARY_PATH && export NCCL_SOCKET_IFNAME=bond0 && export NCCL_BUFFSIZE=2097152 && export TORCH_DIST_INIT_BARRIER=1 && export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 && export HF_HOME=/tmp/hf_cache && export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && export NCCL_NVTX_ENABLE=1 && export TORCH_NCCL_TRACE_BUFFER_SIZE=1000 && export TORCH_NCCL_DUMP_ON_TIMEOUT=1"

TRAIN_PIDS=()

# Master node: nsys wrapping torchrun
node="${NODES[0]}"
echo "  Launching master ${node} with nsys..."
ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && ${ENV_VARS} && nsys profile --trace=cuda,nvtx,cublas,cudnn,osrt --cuda-memory-usage=true --stats=true --force-overwrite=true --output=${OUTPUT_DIR}/nsys/master_${node} --sample=none ${TORCHRUN_CMD} 2>&1 | tee ${OUTPUT_DIR}/nsys/master_${node}_stdout.log" &
TRAIN_PIDS+=($!)

# Worker nodes
for i in $(seq 1 $((NNODES - 1))); do
    node="${NODES[$i]}"
    echo "  Launching worker ${node}..."
    ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && ${ENV_VARS} && ${TORCHRUN_CMD} 2>&1 | tee ${OUTPUT_DIR}/worker_${node}_stdout.log" &
    TRAIN_PIDS+=($!)
done

echo ""
echo "[Phase 2] All ${NNODES} nodes launched. Waiting for completion..."
echo ""

TRAIN_EXIT=0
for pid in "${TRAIN_PIDS[@]}"; do
    wait "$pid" || TRAIN_EXIT=$?
done

echo ""
echo "[Phase 2] Training completed (exit code: $TRAIN_EXIT)"

# =============================================================================
# Phase 3: Stop monitoring, collect post-training stats
# =============================================================================
echo ""
echo "[Phase 3] Stopping monitoring and collecting post-training stats..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "if [ -f /tmp/dmon_pid_${SLURM_JOB_ID} ]; then kill \$(cat /tmp/dmon_pid_${SLURM_JOB_ID}) 2>/dev/null; rm -f /tmp/dmon_pid_${SLURM_JOB_ID}; fi; nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv > ${OUTPUT_DIR}/gpu_health/node${i}_${node}_post_training.csv 2>&1" &
done
wait
echo "[Phase 3] Done."

# =============================================================================
# Phase 4: Generate nsys statistics
# =============================================================================
echo ""
echo "[Phase 4] Generating nsys statistics..."
NSYS_REP="${OUTPUT_DIR}/nsys/master_${NODES[0]}.nsys-rep"
if [ -f "$NSYS_REP" ]; then
    NSYS_SIZE=$(du -sh "$NSYS_REP" | cut -f1)
    echo "  nsys report: $NSYS_REP ($NSYS_SIZE)"

    nsys stats --report cuda_gpu_kern_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/cuda_kernel_summary.txt" 2>&1 || true
    nsys stats --report cuda_api_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/cuda_api_summary.txt" 2>&1 || true
    nsys stats --report nvtx_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/nvtx_summary.txt" 2>&1 || true
    nsys stats --report osrt_sum "$NSYS_REP" > "${OUTPUT_DIR}/nsys/osrt_summary.txt" 2>&1 || true

    echo "  Statistics generated."
else
    echo "  WARNING: nsys report not found at $NSYS_REP"
    ls -la "${OUTPUT_DIR}/nsys/" 2>/dev/null || true
fi

# =============================================================================
# Phase 5: Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PROFILING COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""

echo "Artifacts:"
echo ""

echo "1. PyTorch profiler traces:"
find ${OUTPUT_DIR} -name "*trace.json" -type f 2>/dev/null | head -10
echo ""

echo "2. Memory snapshots:"
find ${OUTPUT_DIR} -name "*memory_snapshot*" -type f 2>/dev/null | head -10
echo ""

echo "3. nsys reports:"
ls -lh ${OUTPUT_DIR}/nsys/*.nsys-rep 2>/dev/null || echo "   (none)"
echo ""

echo "4. nsys statistics:"
ls -lh ${OUTPUT_DIR}/nsys/*_summary.txt 2>/dev/null || echo "   (none)"
echo ""

echo "5. GPU metrics (nvidia-smi dmon, ${NNODES} nodes):"
ls -lh ${OUTPUT_DIR}/gpu_metrics/*.csv 2>/dev/null | head -3
echo "   ..."
echo ""

echo "6. GPU health (pre/post training):"
ls ${OUTPUT_DIR}/gpu_health/*.csv 2>/dev/null | head -4
echo "   ..."
echo ""

echo "============================================================"
