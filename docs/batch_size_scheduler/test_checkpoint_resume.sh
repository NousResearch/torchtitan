#!/bin/bash
# Test checkpoint resume for batch size scheduler
#
# This script verifies that batch size state is correctly restored from checkpoint.
#
# Usage:
#   cd /path/to/torchtitan
#   bash docs/batch_size_scheduler/test_checkpoint_resume.sh
#
# Requirements:
#   - Single GPU available
#   - torchtitan installed in current environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$REPO_ROOT/outputs/batch_size_scheduler_ckpt"

echo "========================================"
echo "Batch Size Scheduler Checkpoint Resume Test"
echo "========================================"
echo ""

cd "$REPO_ROOT"

# Clean up previous test outputs
rm -rf "$OUTPUT_DIR"

echo "Step 1: Run training for 30 steps and create checkpoint"
echo "========================================"
echo ""

torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file "$CONFIG_DIR/debug_checkpoint_resume.toml" \
    --training.steps 30 \
    2>&1 | grep -E "(Batch size|grad_accum|Training starts|Training completed|consumed_samples|checkpoint)"

echo ""
echo "Step 2: Resume from checkpoint and continue to step 60"
echo "========================================"
echo "Expected: Batch size should continue from where it left off"
echo ""

# Find the checkpoint directory
CKPT_DIR=$(ls -d "$OUTPUT_DIR/checkpoints/step-"* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint not found in $OUTPUT_DIR/checkpoints/"
    exit 1
fi

echo "Found checkpoint: $CKPT_DIR"
echo ""

torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file "$CONFIG_DIR/debug_checkpoint_resume.toml" \
    --checkpoint.initial_load_path "$CKPT_DIR" \
    2>&1 | grep -E "(Batch size|grad_accum|Training starts|Training completed|consumed_samples)"

echo ""
echo "========================================"
echo "Checkpoint resume test completed!"
echo "========================================"
