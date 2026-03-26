#!/bin/bash
# Benchmark Qwen3 30B-A3B with and without CPU offloading
# Measures: peak memory, tokens/sec across different offload configs
#
# Usage: bash scripts/benchmark_offload.sh
# Requires: 8 GPUs, torchtitan env

set -euo pipefail

CONFIG="torchtitan/models/qwen3/train_configs/qwen3_30b_a3b_offload_bench.toml"
STEPS=15  # warmup + measurement
RESULTS_DIR="/tmp/offload_bench_results"
mkdir -p "$RESULTS_DIR"

PYTHON="/home/phuc/miniconda3/envs/torchtitan/bin/python"
TORCHRUN="/home/phuc/miniconda3/envs/torchtitan/bin/torchrun"

echo "=============================================="
echo "Qwen3 30B-A3B Offload Benchmark"
echo "EP=8, 8 GPUs, seq_len=4096, local_batch=2"
echo "=============================================="

run_experiment() {
    local name="$1"
    local extra_args="$2"
    local log_file="$RESULTS_DIR/${name}.log"

    echo ""
    echo ">>> Running: $name"
    echo "    Args: $extra_args"
    echo "    Log: $log_file"

    $TORCHRUN \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:0" \
        -m torchtitan.train \
        --job.config_file "$CONFIG" \
        --training.steps "$STEPS" \
        $extra_args \
        2>&1 | tee "$log_file"

    # Extract key metrics from log
    echo ""
    echo "--- Results: $name ---"
    grep -E "(peak_memory|tok/s|tps|memory|wps|Memory)" "$log_file" | tail -5 || true
    echo ""
}

# Experiment 1: Baseline (no offloading)
run_experiment "baseline_no_offload" \
    "--training.enable_cpu_offload false"

# Experiment 2: FSDP CPU offload (params + optimizer)
run_experiment "fsdp_cpu_offload" \
    "--training.enable_cpu_offload true"

# Experiment 3: FSDP CPU offload + selective activation checkpoint
run_experiment "fsdp_offload_selective_ac" \
    "--training.enable_cpu_offload true --activation_checkpoint.mode selective"

# Experiment 4: No FSDP offload, full activation checkpoint (baseline for AC comparison)
run_experiment "full_ac_no_offload" \
    "--training.enable_cpu_offload false --activation_checkpoint.mode full"

echo ""
echo "=============================================="
echo "All experiments complete. Results in: $RESULTS_DIR"
echo "=============================================="
echo ""

# Summary table
echo "| Experiment | Log |"
echo "|---|---|"
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    echo "| $name | $f |"
done
