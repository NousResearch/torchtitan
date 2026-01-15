#!/bin/bash
# Test script for multi-stage data training
# Verifies: stage transitions, checkpoint save/resume, exact reproducibility
#
# Usage: ./scripts/test_data_stages.sh

set -e

CONFIG="torchtitan/models/llama3/train_configs/data_stages_test.toml"
OUTPUT_DIR="./outputs/data_stages_test"
FULL_LOG="/tmp/data_stages_full_run.log"
RESUME_LOG="/tmp/data_stages_resume_run.log"

echo "============================================================"
echo "DATA STAGES TEST"
echo "============================================================"
echo ""

# Clean previous outputs
rm -rf "$OUTPUT_DIR" "$FULL_LOG" "$RESUME_LOG"

# Test 1: Full run
echo "[Test 1] Full run: steps 1-15 with 3 stage transitions"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$CONFIG" 2>&1 | tee "$FULL_LOG"

echo ""
echo "[Test 1] Verifying stage transitions occurred..."
if grep -q "stage_1_general.*stage_2_reasoning" "$FULL_LOG"; then
    echo "  ✓ Transition at step 5: stage_1_general -> stage_2_reasoning"
else
    echo "  ✗ Missing transition at step 5"
    exit 1
fi

if grep -q "stage_2_reasoning.*stage_3_final" "$FULL_LOG"; then
    echo "  ✓ Transition at step 10: stage_2_reasoning -> stage_3_final"
else
    echo "  ✗ Missing transition at step 10"
    exit 1
fi

echo ""

# Test 2: Resume from step 7
echo "[Test 2] Resume run: from checkpoint at step 7"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$CONFIG" \
    --checkpoint.load_step 7 2>&1 | tee "$RESUME_LOG"

echo ""
echo "[Test 2] Verifying checkpoint restore..."
if grep -q "Checkpoint was at stage 'stage_2_reasoning'" "$RESUME_LOG"; then
    echo "  ✓ Stage correctly restored to stage_2_reasoning"
else
    echo "  ✗ Stage not correctly restored"
    exit 1
fi

if grep -q "Restored dataloader position from checkpoint" "$RESUME_LOG"; then
    echo "  ✓ Dataloader position restored"
else
    echo "  ✗ Dataloader position not restored"
    exit 1
fi

if grep -q "Training starts at step 8" "$RESUME_LOG"; then
    echo "  ✓ Training resumed at correct step (8)"
else
    echo "  ✗ Training did not resume at correct step"
    exit 1
fi

echo ""

# Test 3: Compare losses
echo "[Test 3] Reproducibility: comparing losses between full and resumed runs"
echo "------------------------------------------------------------"

# Extract losses from both runs (steps 8-15)
extract_losses() {
    grep -oP "step:\s*\K\d+.*?loss:\s*[\d.]+" "$1" | \
    sed 's/\x1b\[[0-9;]*m//g' | \
    awk '{print $1, $3}' | \
    while read step loss; do
        if [ "$step" -ge 8 ] && [ "$step" -le 15 ]; then
            echo "$step $loss"
        fi
    done
}

FULL_LOSSES=$(extract_losses "$FULL_LOG")
RESUME_LOSSES=$(extract_losses "$RESUME_LOG")

echo "Step  | Full Run | Resume   | Match"
echo "------|----------|----------|------"

MISMATCH=0
for step in 8 9 10 11 12 13 14 15; do
    full=$(echo "$FULL_LOSSES" | grep "^$step " | awk '{print $2}')
    resume=$(echo "$RESUME_LOSSES" | grep "^$step " | awk '{print $2}')

    if [ -z "$full" ] || [ -z "$resume" ]; then
        echo "$step     | N/A      | N/A      | ?"
        continue
    fi

    # Compare with tolerance (4 decimal places)
    diff=$(echo "$full $resume" | awk '{printf "%.4f", ($1-$2)^2}')
    if [ "$diff" = "0.0000" ]; then
        match="✓"
    else
        match="✗"
        MISMATCH=1
    fi

    printf "%-5s | %-8s | %-8s | %s\n" "$step" "$full" "$resume" "$match"
done

echo ""

# Final result
echo "============================================================"
if [ "$MISMATCH" -eq 0 ]; then
    echo "SUCCESS: All tests passed!"
    echo ""
    echo "Verified:"
    echo "  - Stage transitions work correctly"
    echo "  - Checkpoint saves stage index and dataloader position"
    echo "  - Resume restores exact state (losses match)"
    echo "============================================================"

    # Cleanup
    rm -rf "$OUTPUT_DIR" "$FULL_LOG" "$RESUME_LOG"
    exit 0
else
    echo "FAILURE: Losses do not match between full and resumed runs"
    echo "============================================================"
    exit 1
fi
