#!/bin/bash
# Start vLLM Inference Server

set -e

echo "========================================"
echo "Starting vLLM Inference Server"
echo "========================================"

# Cleanup function
cleanup_vllm() {
    echo "Cleaning up any existing vLLM processes..."
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "torchtitan.grpo.vllm_handling.vllm_runner" 2>/dev/null || true
    lsof -ti:9001 | xargs kill -9 2>/dev/null || true
    sleep 2
}

cleanup_vllm

# Use the separate vLLM environment (not the training env)
source /home/nightwing/envs/vllm/.venv/bin/activate

# Configuration
MODEL_PATH="/home/nightwing/Projects/torchtitan/tmp/qwen3-1.7b-hf"  # HF checkpoint path or name
TP_SIZE=1
BASE_PORT=9001

# Set LOGDIR to match the trainer (needed for distributed_updater coordination)
export LOGDIR="${LOGDIR:-/tmp/torchtitan_logs}"
mkdir -p "$LOGDIR"

# Set NUM_INFERENCE_NODES=0 for single node setup (required by distributed_updater)
export NUM_INFERENCE_NODES=0

echo "Starting vLLM server on port $BASE_PORT..."
echo "CUDA_VISIBLE_DEVICES: 4"
echo "Model: $MODEL_PATH"
echo "LOGDIR: $LOGDIR"
echo "NUM_INFERENCE_NODES: $NUM_INFERENCE_NODES"

# Run vLLM on GPU 4 (training uses GPUs 0-3)
# IMPORTANT: Set CUDA_VISIBLE_DEVICES as prefix, not export
CUDA_VISIBLE_DEVICES=4 nohup python -m torchtitan.grpo.vllm_handling.vllm_runner \
    --model "$MODEL_PATH" \
    --port $BASE_PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.75 \
    --dtype="bfloat16" \
    --log-level="error" \
    > "${LOGDIR}/vllm_${BASE_PORT}.log" 2>&1 &

SERVER_PID=$!
echo "vLLM server starting (PID: $SERVER_PID)"

echo ""
echo "Waiting for server to be ready (~30 seconds)..."
sleep 60

echo ""
echo "Testing server connectivity..."
if curl -s "http://localhost:${BASE_PORT}/health" > /dev/null; then
    echo "✓ vLLM server on port $BASE_PORT is ready"
else
    echo "✗ vLLM server on port $BASE_PORT is not responding"
    echo "Check log at: /tmp/vllm_server_${BASE_PORT}.log"
    echo "Last 20 lines of log:"
    tail -20 "/tmp/vllm_server_${BASE_PORT}.log" 2>/dev/null || echo "Log file not found"
fi

echo ""
echo "vLLM server ready for inference!"
echo "Log available at: /tmp/vllm_server_${BASE_PORT}.log"
