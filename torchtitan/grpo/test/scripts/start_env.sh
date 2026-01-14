#!/bin/bash
# Start GSM8k Environment Server

set -e

echo "========================================"
echo "Starting GSM8k Environment Server"
echo "========================================"

# Add Atropos to PYTHONPATH
export PYTHONPATH=/home/shared/atropos:$PYTHONPATH

# Get the torchtitan root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "TorchTitan root: $TORCHTITAN_ROOT"
cd "$TORCHTITAN_ROOT"

# Configuration
MODEL_NAME="Qwen/Qwen3-1.7B"
SGLANG_URL_1="http://localhost:9001/v1"
SGLANG_URL_2="http://localhost:9002/v1"

# Check if Atropos is accessible
if ! python -c "from atroposlib.envs.base import BaseEnv" 2>/dev/null; then
    echo "ERROR: Cannot import Atropos. Is PYTHONPATH set correctly?"
    echo "PYTHONPATH=$PYTHONPATH"
    exit 1
fi

# Check if SGLang servers are running
echo "Checking SGLang server availability..."
if ! curl -s "$SGLANG_URL_1/models" > /dev/null; then
    echo "WARNING: SGLang server at $SGLANG_URL_1 is not responding"
fi
if ! curl -s "$SGLANG_URL_2/models" > /dev/null; then
    echo "WARNING: SGLang server at $SGLANG_URL_2 is not responding"
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
