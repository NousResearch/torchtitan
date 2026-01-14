#!/bin/bash
# Master script to launch the full RL test pipeline

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "GSM8k RL Test Pipeline Launcher"
echo "========================================"
echo ""
echo "This script will start all components:"
echo "  1. Atropos API Server"
echo "  2. SGLang Inference Servers"
echo "  3. GSM8k Environment Server"
echo "  4. TorchTitan Trainer"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "========================================"
    echo "Shutting down all services..."
    echo "========================================"

    if [ ! -z "$API_PID" ]; then
        echo "Stopping Atropos API (PID: $API_PID)"
        kill $API_PID 2>/dev/null || true
    fi

    if [ ! -z "$SGLANG_PID" ]; then
        echo "Stopping SGLang servers (PID: $SGLANG_PID)"
        kill $SGLANG_PID 2>/dev/null || true
        # Also kill any remaining sglang processes
        pkill -f "sglang.launch_server" 2>/dev/null || true
    fi

    if [ ! -z "$ENV_PID" ]; then
        echo "Stopping GSM8k environment (PID: $ENV_PID)"
        kill $ENV_PID 2>/dev/null || true
    fi

    echo "All services stopped"
    exit 0
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Step 1: Start Atropos API
echo "Step 1/4: Starting Atropos API Server..."
"$SCRIPT_DIR/start_api.sh" > /tmp/atropos_api.log 2>&1 &
API_PID=$!
echo "API started (PID: $API_PID, log: /tmp/atropos_api.log)"
echo "Waiting for API to be ready..."
sleep 5

# Check if API is running
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "ERROR: Atropos API failed to start"
    echo "Check log at: /tmp/atropos_api.log"
    exit 1
fi
echo "API is ready"
echo ""

# Step 2: Start SGLang servers
echo "Step 2/4: Starting SGLang Inference Servers..."
"$SCRIPT_DIR/start_sglang.sh" > /tmp/sglang_launcher.log 2>&1 &
SGLANG_PID=$!
echo "SGLang launcher started (PID: $SGLANG_PID)"
echo "Waiting for SGLang servers to load models (this may take ~30 seconds)..."
sleep 35

# Check if SGLang servers are running
SGLANG_READY=true
for PORT in 9001 9002; do
    if ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
        echo "WARNING: SGLang server on port $PORT is not responding"
        SGLANG_READY=false
    fi
done

if [ "$SGLANG_READY" = false ]; then
    echo "WARNING: Some SGLang servers may not be ready"
    echo "Check logs at: /tmp/sglang_server_*.log"
    echo "Continuing anyway..."
else
    echo "SGLang servers are ready"
fi
echo ""

# Step 3: Start GSM8k environment
echo "Step 3/4: Starting GSM8k Environment Server..."
"$SCRIPT_DIR/start_env.sh" > /tmp/gsm8k_env_wrapper.log 2>&1 &
ENV_PID=$!
echo "Environment started (PID: $ENV_PID, log: /tmp/gsm8k_env.log)"
echo "Waiting for environment to register..."
sleep 5
echo "Environment should be running"
echo ""

# Step 4: Start trainer
echo "Step 4/4: Starting TorchTitan Trainer..."
echo "========================================"
echo ""
"$SCRIPT_DIR/start_trainer.sh"

# If we get here, training completed successfully
echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
echo ""
echo "Logs available at:"
echo "  - Atropos API: /tmp/atropos_api.log"
echo "  - SGLang servers: /tmp/sglang_server_*.log"
echo "  - GSM8k environment: /tmp/gsm8k_env.log"
echo "  - Trainer: $LOGDIR"
echo ""
