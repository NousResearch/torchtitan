#!/bin/bash
#SBATCH --job-name=cp16_lbs8_workenv
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/phuc/004_kimi_1t_upstream_reference/torchtitan/outputs/cp_sweep_local/cp16_lbs8_workenv_%j.out
#SBATCH --error=/home/phuc/004_kimi_1t_upstream_reference/torchtitan/outputs/cp_sweep_local/cp16_lbs8_workenv_%j.err
#SBATCH --exclusive
#SBATCH --exclude=7da1cfaf-13

echo "=============================================="
echo "CP Sweep: 16N EP=128 CP=16 LBS=8"
echo "Using WORKING environment from job 2754"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodelist: $SLURM_NODELIST"
echo "Start: $(date)"

# Setup distributed training environment
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29532
export WORLD_SIZE=$((SLURM_NNODES * 8))

# Environment variables
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export LOGLEVEL=INFO
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# NCCL settings
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_SOCKET_IFNAME=enp90s0f0np0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IGNORE_CPU_AFFINITY=1

# PyTorch allocator settings
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.5,roundup_power2_divisions:16"

# Disable wandb
export WANDB_MODE=disabled
export HF_HOME="/home/phuc/.cache/huggingface"

# ============================================
# WORKING ENVIRONMENT (from job 2754)
# ============================================
export PATH="/home/phuc/kimi_1t_upstream/torchtitan/.venv/bin:$PATH"

# Config file
CONFIG_FILE="/home/phuc/worklogs/2026-02-02/cp_sweep/configs/exp_16n_ep128_cp16_lbs8.toml"

# Change to working torchtitan directory
cd /home/phuc/kimi_1t_upstream/torchtitan

# Print environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Torchtitan path: $(pwd)"
echo "Config: $CONFIG_FILE"
echo ""

# Run training
srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c 'export LOCAL_RANK=$SLURM_LOCALID; export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID)); python -m torchtitan.train --job.config_file '"${CONFIG_FILE}"''

echo "End: $(date)"
