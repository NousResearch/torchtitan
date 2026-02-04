#!/bin/bash
#SBATCH --job-name=bf3_rerun
#SBATCH --nodes=12
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/bf3_rerun_%j.out
#SBATCH --error=logs/bf3_rerun_%j.err
#SBATCH --exclusive

mkdir -p logs
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000

cd /home/phuc/kimi_1t/torchtitan
srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c 'export LOCAL_RANK=$SLURM_LOCALID; export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID)); python -m torchtitan.train --job.config_file torchtitan/models/deepseek_v3/train_configs/exp1bf3_full_bf16_compile.toml'
