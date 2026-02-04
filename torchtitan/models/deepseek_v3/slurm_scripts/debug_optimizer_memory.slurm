#!/bin/bash
#SBATCH --job-name=debug_optim_mem
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_optim_mem_%j.out
#SBATCH --error=logs/debug_optim_mem_%j.err
#SBATCH --exclusive

# =============================================================================
# DEBUG: Verify CPU Memory Exhaustion Hypothesis for optimizer.step() Hang
# =============================================================================
#
# This script runs training WITH optimizer step enabled and comprehensive
# monitoring to verify that the "hang" is caused by CPU memory exhaustion.
#
# Expected observations:
# - RAM usage will climb to ~1.8-1.9 TB during first optimizer.step()
# - Swap in/out rates will spike (swap thrashing)
# - Training will appear to "hang" due to swap thrashing
#
# Key metrics to watch in the CSV output:
# - ram_used_gb: Should climb to ~1850 GB
# - swap_in_mb_s / swap_out_mb_s: High values = thrashing
# - swap_percent: Will hit 100%
# =============================================================================

set -euxo pipefail

echo "=========================================="
echo "DEBUG: Optimizer Memory Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Create log directory
mkdir -p logs
LOG_DIR="logs/${SLURM_JOB_ID}_optimizer_memory_debug"
mkdir -p "$LOG_DIR"

# Get head node for distributed training
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -n 1)
echo "Head node: $HEAD_NODE ($HEAD_ADDR)"

# Environment setup
export MASTER_ADDR=$HEAD_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=12
export NCCL_DEBUG=WARN  # Set to INFO for detailed NCCL logging if needed

# Path setup
cd /home/phuc/kimi_1t/torchtitan
CONFIG="torchtitan/models/deepseek_v3/train_configs/kimi_1t_debug_optimizer_memory.toml"
MONITOR_SCRIPT="profiling_kimi_1t/scripts/comprehensive_resource_monitor.py"

# =============================================================================
# START MONITORING ON ALL NODES
# =============================================================================
echo "Starting comprehensive monitoring on all nodes..."

# Start monitoring on each node (run in background via srun)
for node in $(scontrol show hostnames $SLURM_NODELIST); do
    echo "Starting monitor on $node..."
    srun --nodes=1 --ntasks=1 -w "$node" \
        python "$MONITOR_SCRIPT" \
            --output "$LOG_DIR/${node}_resources.csv" \
            --interval 1 \
            --verbose &
done
sleep 5  # Give monitors time to start

echo "Monitors started. Logs will be in: $LOG_DIR/"

# =============================================================================
# ALSO CAPTURE SYSTEM-LEVEL METRICS
# =============================================================================
# vmstat on head node (captures swap activity)
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" vmstat -t 1 > "$LOG_DIR/vmstat.log" 2>&1 &
VMSTAT_PID=$!

# sar for historical analysis (if available)
if command -v sar &> /dev/null; then
    srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" sar -r -S 1 > "$LOG_DIR/sar_memory.log" 2>&1 &
    SAR_PID=$!
fi

# =============================================================================
# PRE-TRAINING MEMORY SNAPSHOT
# =============================================================================
echo ""
echo "=== PRE-TRAINING MEMORY STATUS ==="
echo "Head node ($HEAD_NODE):"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" free -h
echo ""
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|Cached|Buffers"
echo ""

# =============================================================================
# RUN TRAINING
# =============================================================================
echo ""
echo "=== STARTING TRAINING ==="
echo "Config: $CONFIG"
echo "Expecting: RAM usage to climb to ~1.8TB during optimizer.step()"
echo ""

# Run training with timeout (in case it hangs, we still get the logs)
timeout 900 srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --gpus-per-node=8 \
    bash -c '
        export LOCAL_RANK=$SLURM_LOCALID
        export RANK=$((SLURM_NODEID * 8 + SLURM_LOCALID))

        # Print memory before training on rank 0
        if [ $RANK -eq 0 ]; then
            echo "[Rank 0] Starting training..."
            echo "[Rank 0] Initial memory: $(free -h | grep Mem | awk "{print \$3\"/\"\$2}")"
        fi

        python -m torchtitan.train --job.config_file '"$CONFIG"'
    ' || {
    EXIT_CODE=$?
    echo ""
    echo "=== TRAINING EXITED WITH CODE $EXIT_CODE ==="
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Training timed out (likely hanging due to swap thrashing)"
    fi
}

# =============================================================================
# POST-TRAINING MEMORY SNAPSHOT
# =============================================================================
echo ""
echo "=== POST-TRAINING MEMORY STATUS ==="
echo "Head node ($HEAD_NODE):"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" free -h
echo ""

# =============================================================================
# STOP MONITORING
# =============================================================================
echo ""
echo "Stopping monitors..."
# Kill all background jobs
jobs -p | xargs -r kill 2>/dev/null || true
sleep 2

# =============================================================================
# QUICK ANALYSIS
# =============================================================================
echo ""
echo "=== QUICK ANALYSIS ==="
echo ""

# Check if swap was used
echo "Checking for swap activity..."
if [ -f "$LOG_DIR/vmstat.log" ]; then
    echo "Swap activity from vmstat (si=swap in, so=swap out):"
    head -3 "$LOG_DIR/vmstat.log"
    echo "..."
    tail -10 "$LOG_DIR/vmstat.log"
fi

echo ""
echo "Monitor CSV files:"
ls -la "$LOG_DIR/"*.csv 2>/dev/null || echo "No CSV files found"

# Check peak memory from monitor
echo ""
echo "Peak RAM usage from monitors:"
for csv in "$LOG_DIR"/*_resources.csv; do
    if [ -f "$csv" ]; then
        node=$(basename "$csv" _resources.csv)
        peak_ram=$(tail -n +2 "$csv" | cut -d',' -f5 | sort -n | tail -1)
        peak_swap=$(tail -n +2 "$csv" | cut -d',' -f11 | sort -n | tail -1)
        echo "  $node: Peak RAM=${peak_ram}GB, Peak Swap%=${peak_swap}%"
    fi
done

echo ""
echo "=========================================="
echo "DEBUG COMPLETE"
echo "Full logs in: $LOG_DIR/"
echo "=========================================="
echo ""
echo "To analyze:"
echo "  1. Check RAM usage over time: column 'ram_used_gb' in CSV"
echo "  2. Check swap thrashing: columns 'swap_in_mb_s', 'swap_out_mb_s'"
echo "  3. Look for RAM climbing to ~1850GB during optimizer.step()"
echo ""
