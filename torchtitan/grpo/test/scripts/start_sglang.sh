#!/bin/bash
# Start SGLang Inference Server

set -e

echo "========================================"
echo "Starting SGLang Inference Server"
echo "========================================"

# Cleanup function
cleanup_sglang() {
    echo "Cleaning up any existing SGLang processes..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
}

# Run cleanup first
cleanup_sglang

source /home/nightwing/Projects/torchtitan/sglangvenv/bin/activate

# Configuration
MODEL_PATH="/home/nightwing/Projects/torchtitan/tmp/qwen3-1.7b-hf"  # HF path
TP_SIZE=1
NUM_SERVERS=1  # Using 1 server for testing to avoid OOM
BASE_PORT=9001

# Note: Using HF model name - SGLang will auto-download if needed

# Start SGLang server
echo "Starting SGLang server on port $BASE_PORT..."

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $BASE_PORT \
    --tp $TP_SIZE \
    --host 0.0.0.0 \
    --log-level info \
    2>&1 | tee "/tmp/sglang_server_${BASE_PORT}.log" &

SERVER_PID=$!
echo "SGLang server starting (PID: $SERVER_PID)"

echo ""
echo "Waiting for server to be ready (~30 seconds)..."
sleep 60

# Test server connectivity
echo ""
echo "Testing server connectivity..."
if curl -s "http://localhost:${BASE_PORT}/v1/models" > /dev/null; then
    echo "Server on port $BASE_PORT is ready"
else
    echo "Server on port $BASE_PORT is not responding"
    echo "Check log at: /tmp/sglang_server_${BASE_PORT}.log"
fi

echo ""
echo "SGLang server ready for inference!"
echo "Log available at: /tmp/sglang_server_${BASE_PORT}.log"
