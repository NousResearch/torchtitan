#!/bin/bash
#SBATCH --job-name=12n_cp2_32768_fixed
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --partition=batch
#SBATCH --output=/home/phuc/kimi_1t/slurm_logs/12n_cp2_32768_fixed-%j.out

echo "Starting: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Testing 32768 context with 12 nodes EP=96 CP=2 (with torch.compile fix)"

cd /home/phuc/kimi_1t/torchtitan

# Activate environment
source /home/phuc/kimi_1t/env/bin/activate

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node IP: $head_node_ip"

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

CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/exp1a/exp1ab/exp1ab10_12n_EP96_CP2_LBS1_ctx32k.toml"

srun torchrun \
    --nnodes 12 \
    --nproc_per_node 8 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file ${CONFIG_FILE}

echo "Finished: $(date)"
