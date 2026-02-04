#!/bin/bash
#SBATCH --job-name=debug_nan_tracker
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=outputs/debug_cp_nan/slurm_%j.out
#SBATCH --error=outputs/debug_cp_nan/slurm_%j.err

set -ex

# Setup environment
cd /home/phuc/kimi_1t/torchtitan
export PATH=/home/phuc/kimi_1t/env/bin:$PATH

# Get head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"

# Environment settings
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# Config file
CONFIG_FILE=torchtitan/models/deepseek_v3/train_configs/debug_cp_nan/debug_kimi_k2_8n_EP64_CP2_no_tracker.toml

# Create output directory
mkdir -p outputs/debug_cp_nan

# Launch with srun + torchrun (one torchrun per node)
srun bash -c "torchrun \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    -m torchtitan.train \
    --job.config_file $CONFIG_FILE"
