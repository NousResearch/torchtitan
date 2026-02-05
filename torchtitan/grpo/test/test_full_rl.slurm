#!/bin/bash
#SBATCH --job-name=grpo_full_test
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

# Create logs directory
mkdir -p logs/$SLURM_JOB_ID

# Set ulimit higher
ulimit -n 32000
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"

# echo slurm nodes
echo "SLURM nodes: $SLURM_JOB_NODELIST"

# Basic config stuff - pointing to test setup
export CONFIG_FILE="$(pwd)/torchtitan/grpo/test/test_config.toml"
export MODEL_NAME="/home/nightwing/Projects/torchtitan/tmp/qwen2.5-7b"
export PYTHON_SCRIPT="$(pwd)/torchtitan/grpo/test/gsm8k_server.py"
export PYTHON_ARGS=""
export TRAINING_ARGS=""
export NUM_TRAINING_NODES=1
export NUM_INFERENCE_NODES=1

# NCCL settings
export NCCL_BUFFSIZE=33554432
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IGNORE_CPU_AFFINITY=1

# Define environment paths
export TRAIN_PATH="$(pwd)"
export TRAIN_ENV="/home/nightwing/miniconda3/envs/torchtitan/"
export VLLM_ENV="/home/nightwing/miniconda3/envs/vllm/"
export API_ENV="${TRAIN_ENV}"

# Get head node info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Use the proven working vllm_launch.sh script
srun -l --export=ALL ./vllm_launch.sh
