#!/bin/bash
#SBATCH --job-name=kimi_k2_ep64_cp1_8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=./outputs/kimi_k2_8n_ep64_cp1_%j_%N.out
#SBATCH --error=./outputs/kimi_k2_8n_ep64_cp1_%j_%N.err
#SBATCH --exclusive
# -003 missing gpu, -020 iband error, -032 giving nccl errors, -064 acting as head node
#SBATCH --exclude=sfc-003,sfc-028

export NCCL_CONF_FILE=/etc/nccl.conf
export NCCL_TIMEOUT=1800000

export NCCL_DEBUG=WARN
export LOGLEVEL=INFO

# Use local /tmp for HF cache to avoid NFS filelock issues
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export TRITON_HOME=/tmp/tritoncache
export NUMBA_CACHE_DIR=/tmp/numbacache
export OMP_NUM_THREADS=4

# enable conda environment properly
export CONDA_PREFIX=/mnt/data/home/nous/miniconda3
. "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate /mnt/data/home/nous/phuc/kimi_k2_pretraining/.venv

# CUDA and library setup
export CUDA_HOME=/mnt/data/home/nous/phuc/kimi_k2_pretraining/cuda-12.8/unified
export NVSHMEM_DIR=/mnt/data/home/nous/phuc/kimi_k2_pretraining/.venv/lib/python3.10/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$NVSHMEM_DIR/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH=$CUDA_HOME/bin:$NVSHMEM_DIR/bin:$PATH

# Set the RDZV_HOST variable to the hostname of the node running the script
export RDZV_HOST=$(hostname -s)
export RDZV_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

CONFIG_FILE="/mnt/data/home/nous/phuc/kimi_k2_pretraining/torchtitan/torchtitan/models/deepseek_v3/train_configs/kimi_k2_ep64_cp1_seq24k_lbs1_8n.toml"
cd /mnt/data/home/nous/phuc/kimi_k2_pretraining/torchtitan

srun torchrun \
    --nnodes "${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node 8 \
    --rdzv_id "${SLURM_JOB_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${RDZV_HOST}:${RDZV_PORT}" \
    --log-dir "/mnt/data/home/nous/phuc/kimi_k2_pretraining/torchtitan/torchrun/${SLURM_JOB_NAME}" \
    --tee 3 \
    -m torchtitan.train \
    --job.config_file "${CONFIG_FILE}" \
    "$@"
