#!/bin/bash
# Start GSM8k Environment Server

set -e

echo "========================================"
echo "Starting GSM8k Environment Server"
echo "========================================"

# Activate TorchTitan venv (has atroposlib installed)
source /home/nightwing/Projects/torchtitan/.venv/bin/activate

# Get the torchtitan root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "TorchTitan root: $TORCHTITAN_ROOT"
cd "$TORCHTITAN_ROOT"

# Configuration
MODEL_NAME="Qwen/Qwen3-1.7B"
VLLM_URL="http://localhost:9001/v1"

# Check if Atropos is accessible
if ! python -c "from atroposlib.envs.base import BaseEnv" 2>/dev/null; then
    echo "ERROR: Cannot import Atropos. Is it installed in the venv?"
    echo "Run: pip install -e /home/shared/atropos"
    exit 1
fi

# Check if vLLM server is running
echo "Checking vLLM server availability..."
if ! curl -s "$VLLM_URL/models" > /dev/null; then
    echo "WARNING: vLLM server at $VLLM_URL is not responding"
fi

# Check if Atropos API is running
echo "Checking Atropos API availability..."
if ! curl -s "http://localhost:8000/" > /dev/null; then
    echo "ERROR: Atropos API is not running on http://localhost:8000"
    echo "Please start the API server first (./start_api.sh)"
    exit 1
fi

echo ""
echo "Starting GSM8k environment..."
python torchtitan/grpo/test/gsm8k_server.py serve \
    --slurm false \
    --openai.model_name "$MODEL_NAME" \
    2>&1 | tee /tmp/gsm8k_env.log

echo ""
echo "GSM8k environment stopped"
echo "Log available at: /tmp/gsm8k_env.log"
