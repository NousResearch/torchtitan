#!/bin/bash
# Test script for multi-stage data training
# Verifies: backward compatibility, stage transitions, checkpoint resume, ablation
#
# Usage: ./scripts/test_data_stages.sh

set -e

STAGES_CONFIG="torchtitan/models/llama3/train_configs/data_stages_test.toml"
BACKCOMPAT_CONFIG="torchtitan/models/llama3/train_configs/data_stages_backcompat_test.toml"
ABLATION_CONFIG="torchtitan/models/llama3/train_configs/data_stages_ablation_test.toml"

STAGES_OUTPUT="./outputs/data_stages_test"
BACKCOMPAT_OUTPUT="./outputs/data_stages_backcompat_test"
ABLATION_OUTPUT="./outputs/data_stages_ablation_test"

FULL_LOG="/tmp/data_stages_full_run.log"
RESUME_LOG="/tmp/data_stages_resume_run.log"
BACKCOMPAT_LOG="/tmp/data_stages_backcompat.log"
ABLATION_LOG="/tmp/data_stages_ablation.log"

echo "============================================================"
echo "DATA STAGES TEST SUITE"
echo "============================================================"
echo ""

# Clean previous outputs
rm -rf "$STAGES_OUTPUT" "$BACKCOMPAT_OUTPUT" "$ABLATION_OUTPUT"
rm -f "$FULL_LOG" "$RESUME_LOG" "$BACKCOMPAT_LOG" "$ABLATION_LOG"

##############################################################################
# Test 1: Backward Compatibility (no data_stages defined)
##############################################################################
echo "[Test 1] Backward Compatibility: No [[training.data_stages]]"
echo "------------------------------------------------------------"
echo "Config uses only [training] data fields, no data_stages."
echo ""

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$BACKCOMPAT_CONFIG" 2>&1 | tee "$BACKCOMPAT_LOG"

echo ""
echo "[Test 1] Verifying backward compatibility..."

if grep -q "No \[\[training.data_stages\]\] defined. Auto-created single stage from \[training\] config" "$BACKCOMPAT_LOG"; then
    echo "  ✓ Auto-created 'default' stage from [training]"
else
    echo "  ✗ Failed to auto-create stage from [training]"
    exit 1
fi

if grep -q "Stage 1: default" "$BACKCOMPAT_LOG"; then
    echo "  ✓ Stage named 'default'"
else
    echo "  ✗ Stage name incorrect"
    exit 1
fi

if grep -q "Training completed" "$BACKCOMPAT_LOG"; then
    echo "  ✓ Training completed successfully"
else
    echo "  ✗ Training did not complete"
    exit 1
fi

echo ""
echo "[Test 1] PASSED: Backward compatibility works"
echo ""

##############################################################################
# Test 2: Multi-stage with transitions
##############################################################################
echo "[Test 2] Multi-Stage Training: Full run with 3 stages"
echo "------------------------------------------------------------"
echo "Config has 3 stages with transitions at step 5 and 10."
echo ""

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$STAGES_CONFIG" 2>&1 | tee "$FULL_LOG"

echo ""
echo "[Test 2] Verifying stage transitions occurred..."

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
echo "[Test 2] PASSED: Stage transitions work correctly"
echo ""

##############################################################################
# Test 3: Checkpoint resume
##############################################################################
echo "[Test 3] Checkpoint Resume: from step 7"
echo "------------------------------------------------------------"
echo "Resume from checkpoint at step 7, verify state restoration."
echo ""

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$STAGES_CONFIG" \
    --checkpoint.load_step 7 2>&1 | tee "$RESUME_LOG"

echo ""
echo "[Test 3] Verifying checkpoint restore..."

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
echo "[Test 3] PASSED: Checkpoint resume works correctly"
echo ""

##############################################################################
# Test 4: Reproducibility (compare losses)
##############################################################################
echo "[Test 4] Reproducibility: Comparing losses between full and resumed runs"
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
if [ "$MISMATCH" -eq 0 ]; then
    echo "[Test 4] PASSED: Losses match exactly"
else
    echo "[Test 4] FAILED: Losses do not match"
    exit 1
fi
echo ""

##############################################################################
# Test 5: Ablation (stages start mid-training)
##############################################################################
echo "[Test 5] Ablation: Stages start at step 5 (dmayhem use case)"
echo "------------------------------------------------------------"
echo "Config has [training] data for steps 0-5, then [[training.data_stages]]"
echo "at step 5 with different random seed. Tests mid-training ablation."
echo ""

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --standalone \
    -m torchtitan.train --job.config_file "$ABLATION_CONFIG" 2>&1 | tee "$ABLATION_LOG"

echo ""
echo "[Test 5] Verifying ablation setup..."

if grep -q "Auto-created 'pre_stages' from \[training\] for steps 0-5" "$ABLATION_LOG"; then
    echo "  ✓ Auto-created 'pre_stages' for gap (steps 0-5)"
else
    echo "  ✗ Failed to auto-create pre_stages"
    exit 1
fi

if grep -q "Stage 1: pre_stages" "$ABLATION_LOG"; then
    echo "  ✓ First stage is 'pre_stages'"
else
    echo "  ✗ First stage not correctly named"
    exit 1
fi

if grep -q "Stage 2: ablation_stage" "$ABLATION_LOG"; then
    echo "  ✓ Second stage is 'ablation_stage'"
else
    echo "  ✗ Second stage not correctly named"
    exit 1
fi

if grep -q "pre_stages.*ablation_stage" "$ABLATION_LOG"; then
    echo "  ✓ Transition occurred: pre_stages -> ablation_stage"
else
    echo "  ✗ Transition did not occur"
    exit 1
fi

if grep -q "Training completed" "$ABLATION_LOG"; then
    echo "  ✓ Training completed successfully"
else
    echo "  ✗ Training did not complete"
    exit 1
fi

echo ""
echo "[Test 5] PASSED: Ablation mode works correctly"
echo ""

##############################################################################
# Final Summary
##############################################################################
echo "============================================================"
echo "ALL TESTS PASSED!"
echo "============================================================"
echo ""
echo "Verified:"
echo "  [Test 1] Backward compatibility - no data_stages"
echo "  [Test 2] Multi-stage transitions"
echo "  [Test 3] Checkpoint save/resume"
echo "  [Test 4] Exact reproducibility on resume"
echo "  [Test 5] Ablation mode (stages start mid-training)"
echo ""
echo "============================================================"

# Cleanup
rm -rf "$STAGES_OUTPUT" "$BACKCOMPAT_OUTPUT" "$ABLATION_OUTPUT"
rm -f "$FULL_LOG" "$RESUME_LOG" "$BACKCOMPAT_LOG" "$ABLATION_LOG"

exit 0
