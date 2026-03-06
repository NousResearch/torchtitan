#!/bin/bash
#SBATCH --job-name=multinode_online_runner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# set ulimit higher... dang async io
ulimit -n 32000
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"


# echo slurm nodes
echo "SLURM nodes: $SLURM_JOB_NODELIST"

# basic config stuff
export CONFIG_FILE=${CONFIG_FILE:-"/data/home/mormio/torchtitan-grpo/torchtitan/grpo/configs/nous-3ba-30b-scienceqa-debug.toml"}
export MODEL_NAME=${MODEL_NAME:-"/data/home/mormio/models/qwen3_30b_a3b_scienceqa_20260128_ckpt191"}
export PYTHON_SCRIPT=${PYTHON_SCRIPT:-"/home/mormio/AtroposEnvironments/scienceolympiad/science_olympiad_server.py"}
export PYTHON_ARGS=${PYTHON_ARGS:-""}
export TRAINING_ARGS=${TRAINING_ARGS:-""}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-65536}
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.80}
export NUM_TRAINING_NODES=${NUM_TRAINING_NODES:-2}
export NUM_INFERENCE_NODES=${NUM_INFERENCE_NODES:-2}
export NCCL_BUFFSIZE=33554432
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1


# Define environment paths
export TRAIN_PATH=${TRAIN_PATH:-"/data/home/mormio/torchtitan-grpo"}
export TRAIN_ENV=${TRAIN_ENV:-"/data/home/mormio/miniconda3/envs/grpo-train"}
export VLLM_ENV=${VLLM_ENV:-"/data/home/mormio/miniconda3/envs/grpo-vllm"}
export API_ENV=${API_ENV:-"/data/home/mormio/miniconda3/envs/grpo-atropos"}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(getent hosts "$head_node" | awk '{print $1}' | head -1)

# Build INFERENCE_NODE_HOSTS from the last NUM_INFERENCE_NODES nodes
export INFERENCE_NODE_HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -$NUM_INFERENCE_NODES | tr '\n' ',' | sed 's/,$//')
echo "Inference node hosts: $INFERENCE_NODE_HOSTS"

srun -l --export=ALL ./vllm_launch.sh
