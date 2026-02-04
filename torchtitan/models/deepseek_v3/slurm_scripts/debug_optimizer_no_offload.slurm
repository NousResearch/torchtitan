#!/bin/bash
#SBATCH --job-name=debug_no_off
#SBATCH --output=/home/phuc/kimi_1t/slurm_logs/debug_optimizer_no_offload-%j.out
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:15:00

mkdir -p /home/phuc/kimi_1t/slurm_logs

CONFIG_FILE="/home/phuc/kimi_1t/torchtitan/torchtitan/models/deepseek_v3/train_configs/debug_optimizer_no_offload.toml"

cd /home/phuc/kimi_1t/torchtitan

srun bash -c '
source /home/phuc/kimi_1t/env/bin/activate
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1):29500 \
    -m torchtitan.train \
    --job.config_file '"$CONFIG_FILE"'
'
