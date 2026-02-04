#!/bin/bash
#SBATCH --job-name=profile_16n
#SBATCH --ntasks=16
#SBATCH --nodes=16
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=/home/phuc/kimi_1t/torchtitan/outputs/torch_profiler_16n/profile_%j.out

# ============================================================================
# TorchTitan PyTorch Profiler - 16 Nodes (128 GPUs)
# ============================================================================

echo "========================================================================="
echo "TORCHTITAN PROFILING - 16 NODES"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES ($(($SLURM_NNODES * 8)) GPUs total)"
echo "Start time: $(date)"
echo ""

# Setup nodes
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

export TRITON_HOME=/tmp/emotritoncache_$$_$SLURM_PROCID
export NUMBA_CACHE_DIR=/tmp/numbacache_$$_$SLURM_PROCID
mkdir -p $TRITON_HOME $NUMBA_CACHE_DIR

export WANDB_MODE=disabled
export HF_HOME="/home/phuc/.cache/huggingface"
mkdir -p $HF_HOME

# Use local environment
export PATH="/home/phuc/kimi_1t/env/bin:$PATH"
export CONDA_PREFIX="/home/phuc/kimi_1t/env"

# ============================================================================
# Output Directories
# ============================================================================

RESULT_DIR="/home/phuc/kimi_1t/torchtitan/outputs/torch_profiler_16n/run_${SLURM_JOB_ID}"
mkdir -p ${RESULT_DIR}
mkdir -p /home/phuc/kimi_1t/torchtitan/outputs/torch_profiler_16n

# ============================================================================
# Configuration
# ============================================================================

CONFIG_FILE="/home/phuc/kimi_1t/torchtitan/torch_profiler_16n_config.toml"

echo "Config: ${CONFIG_FILE}"
echo "Output: ${RESULT_DIR}"
echo "EP=128, CP=16, LBS=8, seq_len=32768"
echo ""

# Change to torchtitan directory
cd /home/phuc/kimi_1t/torchtitan

# ============================================================================
# Launch Training with Profiling
# ============================================================================

echo "========================================================================="
echo "Starting 16-node profiling run..."
echo "========================================================================="

srun --export=ALL torchrun \
    --nnodes 16 \
    --nproc_per_node 8 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file ${CONFIG_FILE} \
    --job.dump_folder ${RESULT_DIR} \
    2>&1 | tee ${RESULT_DIR}/training_output.log

TRAIN_EXIT=$?

echo ""
echo "Training exit code: ${TRAIN_EXIT}"

# ============================================================================
# Post-Job Analysis
# ============================================================================

echo ""
echo "========================================================================="
echo "PERFORMANCE METRICS"
echo "========================================================================="
grep -E "MFU|tokens/sec|iteration time|wps|step" ${RESULT_DIR}/training_output.log | tail -30

echo ""
echo "========================================================================="
echo "TRACE FILES"
echo "========================================================================="
if [ -d "${RESULT_DIR}/traces" ]; then
    echo "Traces saved to: ${RESULT_DIR}/traces/"
    ls -lh ${RESULT_DIR}/traces/iteration_*/rank0_trace.json 2>/dev/null || echo "  (checking...)"

    echo ""
    echo "To analyze, run:"
    echo "  python /home/phuc/kimi_1t/torchtitan/analyze_cpu_gpu_boundary.py ${RESULT_DIR}/traces/iteration_3/rank0_trace.json"
else
    echo "No trace directory found yet"
fi

echo ""
echo "========================================================================="
echo "PROFILING COMPLETE"
echo "========================================================================="
echo "Results directory: ${RESULT_DIR}"
echo "End time: $(date)"
