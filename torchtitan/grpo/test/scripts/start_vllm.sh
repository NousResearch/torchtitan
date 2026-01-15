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
    sleep 2
}

cleanup_vllm

source /home/nightwing/Projects/torchtitan/.venv/bin/activate

# Configuration
MODEL_PATH="/home/nightwing/Projects/torchtitan/tmp/qwen3-1.7b-hf"  # HF checkpoint path or name
TP_SIZE=1
BASE_PORT=9001

echo "Starting vLLM server on port $BASE_PORT..."
echo "Model: $MODEL_PATH"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $BASE_PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    2>&1 | tee "/tmp/vllm_server_${BASE_PORT}.log" &

SERVER_PID=$!
echo "vLLM server starting (PID: $SERVER_PID)"

echo ""
echo "Waiting for server to be ready (~30 seconds)..."
sleep 60

echo ""
echo "Testing server connectivity..."
if curl -s "http://localhost:${BASE_PORT}/v1/models" > /dev/null; then
    echo "✓ vLLM server on port $BASE_PORT is ready"
else
    echo "✗ vLLM server on port $BASE_PORT is not responding"
    echo "Check log at: /tmp/vllm_server_${BASE_PORT}.log"
fi

echo ""
echo "vLLM server ready for inference!"
echo "Log available at: /tmp/vllm_server_${BASE_PORT}.log"
