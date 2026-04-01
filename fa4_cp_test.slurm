#!/bin/bash
# FA4 + Context Parallel comparison test — 100M model
# Runs both round_robin and striped distributions sequentially
#
# Usage: sbatch fa4_cp_test.slurm
# Override: CP_DEGREE=4 SEQ_LEN=16384 STEPS=50 sbatch fa4_cp_test.slurm

#SBATCH --job-name=fa4_cp_test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --output=fa4_cp_test_%j.log

set -ex

source /home/nous/miniconda3/etc/profile.d/conda.sh
conda activate omni

# --- Environment ---
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export FI_PROVIDER="efa"
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_BUFFSIZE=2097152
export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export HF_HOME="/home/nous/.cache/huggingface"
export WANDB_PROJECT="torchtitan-fa4-cp-test"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

NGPU=8
NNODES=${SLURM_NNODES:-1}

# Override via env vars
CP_DEGREE=${CP_DEGREE:-2}
SEQ_LEN=${SEQ_LEN:-32768}
STEPS=${STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-1}

echo "============================================"
echo "FA4 CP Test (100M): NGPU=$NGPU NNODES=$NNODES CP=$CP_DEGREE SEQ_LEN=$SEQ_LEN STEPS=$STEPS"
echo "============================================"

# --- Test 1: Round-robin ---
echo ""
echo ">>> Running FA4 + CP (round_robin) <<<"
srun torchrun --nnodes $NNODES --nproc_per_node $NGPU --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file ./torchtitan/models/llama3/train_configs/fa4_cp_round_robin.toml \
    --parallelism.context_parallel_degree $CP_DEGREE \
    --training.seq_len $SEQ_LEN \
    --training.steps $STEPS \
    --training.local_batch_size $BATCH_SIZE \
    2>&1 | tee fa4_cp_round_robin.log

echo ""
echo ">>> Running FA4 + CP (striped) <<<"
srun torchrun --nnodes $NNODES --nproc_per_node $NGPU --rdzv_id 102 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file ./torchtitan/models/llama3/train_configs/fa4_cp_striped.toml \
    --parallelism.context_parallel_degree $CP_DEGREE \
    --training.seq_len $SEQ_LEN \
    --training.steps $STEPS \
    --training.local_batch_size $BATCH_SIZE \
    2>&1 | tee fa4_cp_striped.log

echo ""
echo "============================================"
echo "Both runs complete. Compare logs:"
echo "  fa4_cp_round_robin.log"
echo "  fa4_cp_striped.log"
echo "============================================"
