#!/bin/bash
# Test script for batch size scheduler feature
#
# This script runs a series of tests to verify the batch size scheduler works correctly.
#
# Usage:
#   cd /path/to/torchtitan
#   bash docs/batch_size_scheduler/test_batch_size_scheduler.sh
#
# Requirements:
#   - Single GPU available
#   - torchtitan installed in current environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR"

echo "========================================"
echo "Batch Size Scheduler Test Suite"
echo "========================================"
echo "Repository: $REPO_ROOT"
echo ""

cd "$REPO_ROOT"

# Clean up previous test outputs
rm -rf ./outputs/batch_size_scheduler_*

echo ""
echo "========================================"
echo "Test 1: Constant Batch Size (Baseline)"
echo "========================================"
echo "Expected: Batch size stays at 32 throughout training"
echo ""

torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file "$CONFIG_DIR/debug_constant.toml" \
    2>&1 | grep -E "(Batch size|grad_accum|Training starts|Training completed)"

echo ""
echo "========================================"
echo "Test 2: Linear Rampup"
echo "========================================"
echo "Expected: Batch size ramps from 8 to 32 over 500 samples"
echo ""

torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file "$CONFIG_DIR/debug_linear.toml" \
    2>&1 | grep -E "(Batch size|grad_accum|Training starts|Training completed)"

echo ""
echo "========================================"
echo "Test 3: Increment Rampup (Megatron-style)"
echo "========================================"
echo "Expected: Batch size steps 8 -> 16 -> 24 -> 32 over 400 samples"
echo ""

torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file "$CONFIG_DIR/debug_increment.toml" \
    2>&1 | grep -E "(Batch size|grad_accum|Training starts|Training completed)"

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
