#!/bin/bash
# Start SGLang Inference Servers

set -e

echo "========================================"
echo "Starting SGLang Inference Servers"
echo "========================================"

# Configuration
MODEL_PATH="/home/shared/torchtitan-conversions/qwen3_1.7b" # update with model path
TP_SIZE=1
NUM_SERVERS=2
BASE_PORT=9001

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script to point to your Qwen3-1.7B checkpoint"
    exit 1
fi

# Start SGLang servers
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Starting SGLang server $((i+1))/$NUM_SERVERS on port $PORT..."

    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port $PORT \
        --tp $TP_SIZE \
        --host 0.0.0.0 \
        --log-level info \
        2>&1 | tee "/tmp/sglang_server_${PORT}.log" &

    echo "SGLang server $((i+1)) starting (PID: $!)"
done

echo ""
echo "All SGLang servers started!"
echo "Waiting for servers to be ready..."
sleep 30

# Test server connectivity
echo ""
echo "Testing server connectivity..."
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
        echo "Server on port $PORT is ready"
    else
        echo "Server on port $PORT is not responding"
    fi
done

echo ""
echo "SGLang servers are ready for inference!"
echo "Logs available at: /tmp/sglang_server_*.log"
