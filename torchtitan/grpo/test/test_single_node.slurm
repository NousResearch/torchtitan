#!/bin/bash
#SBATCH --job-name=grpo_two_node_test
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

# Basic config stuff
export CONFIG_FILE="$(pwd)/torchtitan/grpo/test/test_config.toml"
export MODEL_NAME="/home/nightwing/Projects/torchtitan/tmp/qwen3-1.7b-hf"
export PYTHON_SCRIPT="$(pwd)/torchtitan/grpo/test/gsm8k_server.py"
export PYTHON_ARGS=""
export TRAINING_ARGS=""
export NUM_TRAINING_NODES=1
export NUM_INFERENCE_NODES=1

# Define environment paths
export TRAIN_PATH="$(pwd)"
export TRAIN_ENV="${TRAIN_PATH}/.venv"
export VLLM_ENV="/home/nightwing/envs/vllm/.venv"
export API_ENV="${TRAIN_ENV}"

# Get head node info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Run the test using the same pattern as vllm_launch.sh
srun -l --export=ALL ./torchtitan/grpo/test/scripts/test_launcher.sh
