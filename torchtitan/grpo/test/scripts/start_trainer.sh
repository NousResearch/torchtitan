#!/bin/bash
# Start TorchTitan RL Trainer
# This pulls batches from Atropos API and trains the model

set -e

echo "========================================"
echo "Starting TorchTitan RL Trainer"
echo "========================================"

# Add Atropos to PYTHONPATH
export PYTHONPATH=/home/shared/atropos:$PYTHONPATH

# Get the torchtitan root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "TorchTitan root: $TORCHTITAN_ROOT"
cd "$TORCHTITAN_ROOT"

# Configuration
CONFIG_FILE="torchtitan/grpo/test/test_config.toml"
NGPU=${NGPU:-4}  # default to 4 GPUs, override with: NGPU=8 ./start_trainer.sh
LOG_RANK=${LOG_RANK:-0}

# Set required environment variables
export LOGDIR="${LOGDIR:-/tmp/torchtitan_logs}"
mkdir -p "$LOGDIR"
echo "Logs will be written to: $LOGDIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if Atropos API is running
echo "Checking Atropos API availability..."
if ! curl -s "http://localhost:8000/" > /dev/null; then
    echo "ERROR: Atropos API is not running on http://localhost:8000"
    echo "Please start the API server first (./start_api.sh)"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  - Config file: $CONFIG_FILE"
echo "  - Number of GPUs: $NGPU"
echo "  - Log directory: $LOGDIR"
echo "  - Log rank filter: $LOG_RANK"
echo ""

# Launch trainer with torchrun
echo "Launching trainer..."
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node=$NGPU \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter $LOG_RANK \
    --role rank \
    --tee 3 \
    -m torchtitan.grpo_train \
    --job.config_file "$CONFIG_FILE"

echo ""
echo "Training completed!"
echo "Check logs at: $LOGDIR"
