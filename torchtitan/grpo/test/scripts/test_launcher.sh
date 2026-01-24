#!/bin/bash
# set -e temporarily disabled to see errors
set -x  # Print commands for debugging

printenv
ulimit -n 32000

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$TORCHTITAN_ROOT"

# Set defaults if not running under SLURM
: "${SLURM_NODEID:=0}"
: "${NUM_TRAINING_NODES:=1}"
: "${NUM_INFERENCE_NODES:=0}"
: "${LOGDIR:=${TORCHTITAN_ROOT}/logs/test_run}"
: "${MODEL_NAME:=/home/nightwing/Projects/torchtitan/tmp/qwen3-1.7b-hf}"
: "${CONFIG_FILE:=${TORCHTITAN_ROOT}/torchtitan/grpo/test/test_config.toml}"
: "${API_ENV:=/home/nightwing/Projects/torchtitan/.venv}"
: "${TRAIN_ENV:=/home/nightwing/Projects/torchtitan/.venv}"
: "${VLLM_ENV:=/home/nightwing/envs/vllm/.venv}"

# Export LOGDIR so child processes can see it
export LOGDIR
export NUM_INFERENCE_NODES
export MODEL_NAME
export CONFIG_FILE

mkdir -p "$LOGDIR"

echo "Starting test at $(date)"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "NUM_TRAINING_NODES: $NUM_TRAINING_NODES"
echo "NUM_INFERENCE_NODES: $NUM_INFERENCE_NODES"
echo "LOGDIR: $LOGDIR"
echo "MODEL_NAME: $MODEL_NAME"

# Start API and environment (always on node 0)
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    echo "Starting API and environment server..."
    source ${API_ENV}/bin/activate

    # Start Atropos API
    cd /home/shared/atropos
    run-api > ${LOGDIR}/api.log 2>&1 &
    cd "$TORCHTITAN_ROOT"

    # Start GSM8k environment server
    python torchtitan/grpo/test/gsm8k_server.py serve --slurm=True --openai.model_name="$MODEL_NAME" > ${LOGDIR}/env_server.log 2>&1 &

    deactivate
    echo "Started API and environment server..."
fi

# Start training (on training nodes)
if [[ "$SLURM_NODEID" -lt "$NUM_TRAINING_NODES" ]]; then
    echo "Setting up training environment..."
    source ${TRAIN_ENV}/bin/activate

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}

    export LOGLEVEL=INFO
    export NCCL_DEBUG=WARN
    export PYTHONFAULTHANDLER=1
    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    export CUDA_LAUNCH_BLOCKING=0

    # Launch trainer (vLLM runs on separate inference node)
    echo "Launching trainer..."
    torchrun --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint="$head_node:29500" --role rank --tee 3 \
        -m torchtitan.grpo_train --job.config_file ${CONFIG_FILE}
# else we're on an inference node
else
    echo "Starting vLLM inference server..."
    source ${VLLM_ENV}/bin/activate

    PORT_BASE=9000
    LOG_OFFSET=$((SLURM_NODEID * 8))

    # Start 8 vLLM instances on GPUs 0-7 (matching dakota's setup)
    for i in {0..7}; do
        GPU_ID=$i
        LOG_ID=$((GPU_ID + LOG_OFFSET))
        PORT=$((PORT_BASE + i))
        echo "Starting vLLM instance on GPU $GPU_ID, port $PORT"
        CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m torchtitan.grpo.vllm_handling.vllm_runner \
          --model "$MODEL_NAME" \
          --host 0.0.0.0 \
          --gpu-memory-utilization 0.75 \
          --dtype="bfloat16" \
          --log-level="error" \
          --port $PORT > ${LOGDIR}/vllm_${LOG_ID}.log 2>&1 &
        sleep 3
    done

    # Wait indefinitely (keep inference node alive)
    wait
fi

echo "Test completed at $(date)"
