#!/bin/bash
#SBATCH --job-name=debug_opt_hang
#SBATCH --output=/home/phuc/kimi_1t/slurm_logs/debug_optimizer_hang-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:10:00

mkdir -p /home/phuc/kimi_1t/slurm_logs

cd /home/phuc/kimi_1t/torchtitan

srun bash -c '
source /home/phuc/kimi_1t/env/bin/activate
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    debug_mesh_topology.py
'
