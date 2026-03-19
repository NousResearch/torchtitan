#!/bin/bash
#SBATCH --job-name=multinode_online_runner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# set ulimit higher... dang async io
ulimit -n 32000
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}"


# echo slurm nodes
echo "SLURM nodes: $SLURM_JOB_NODELIST"

# basic config stuff
export CONFIG_FILE=${CONFIG_FILE:-"/home/mormio/torchtitan-grpo/torchtitan/grpo/configs/qwen3-30b-a3b-instruct-sft-scienceqa.toml"}
export MODEL_NAME=${MODEL_NAME:-"/home/mormio/models/qwen3_30b_a3b_scienceqa_20260128_ckpt191"}
export PYTHON_SCRIPT=${PYTHON_SCRIPT:-"/home/mormio/AtroposEnvironments/scienceolympiad/science_olympiad_server.py"}
export PYTHON_ARGS=${PYTHON_ARGS:-""}
export TRAINING_ARGS=${TRAINING_ARGS:-""}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-65536}
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.80}
export NUM_TRAINING_NODES=2
export NUM_INFERENCE_NODES=2
export NCCL_BUFFSIZE=33554432
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# InfiniBand configuration
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}

export SGLANG_UPDATE_PROC_DEBUG=1
export WANDB_DIR="/data/home/mormio/torchtitan-grpo/outputs/30ba3b-instruct-sftd-scienceqa-rl/wandb"
# export TMPDIR="/data/home/mormio/tmp/wandb_${SLURM_JOB_ID}"
# mkdir -p "$TMPDIR"

# Increase NCCL watchdog timeout so vllm weight_updater doesn't get killed
# while waiting between training steps (which can take many minutes)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200


# Define environment paths
export TRAIN_PATH=${TRAIN_PATH:-"/data/home/mormio/torchtitan-grpo"}
export TRAIN_ENV=${TRAIN_ENV:-"/data/home/mormio/miniconda3/envs/grpotrain2"}
export VLLM_ENV=${VLLM_ENV:-"/data/home/mormio/miniconda3/envs/grpo-vllm"}
export API_ENV=${API_ENV:-"/data/home/mormio/miniconda3/envs/grpo-atropos"}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(getent hosts "$head_node" | awk '{print $1}' | head -1)

# Build INFERENCE_NODE_HOSTS from the last NUM_INFERENCE_NODES nodes
export INFERENCE_NODE_HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -$NUM_INFERENCE_NODES | tr '\n' ',' | sed 's/,$//')
echo "Inference node hosts: $INFERENCE_NODE_HOSTS"

# Judge runs on first inference node, port 9010, tp=2 (GPUs 0+1)
FIRST_INFERENCE_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -$NUM_INFERENCE_NODES | head -1)
export EVAL_JUDGE_BASE_URL="http://${FIRST_INFERENCE_NODE}:9010/v1"
export EVAL_JUDGE_MODEL=${JUDGE_MODEL:-"Qwen/Qwen3.5-27B"}
echo "Judge URL: $EVAL_JUDGE_BASE_URL"

# Build sglang_urls dynamically: rollout vllm on GPUs 2-7 (ports 9002-9007) per inference node
SGLANG_URLS=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -$NUM_INFERENCE_NODES | while read node; do
    for port in 9002 9003 9004 9005 9006 9007; do
        echo "${node}:${port}"
    done
done | tr '\n' ',' | sed 's/,$//')
export TRAINING_ARGS="${TRAINING_ARGS} --grpo.sglang_urls ${SGLANG_URLS} --grpo.sglang_slurm_num_nodes 0"
echo "sglang_urls: $SGLANG_URLS"

srun -l --export=ALL ./vllm_launch.sh
