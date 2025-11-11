#!/bin/bash
# Run profiling for EP configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

EP_DEGREE="${1:-both}"

if [ "$EP_DEGREE" != "ep1" ] && [ "$EP_DEGREE" != "ep2" ] && [ "$EP_DEGREE" != "both" ]; then
    echo "Usage: $0 [ep1|ep2|both]"
    echo ""
    echo "Examples:"
    echo "  $0 ep1      # Run profiling for EP=1 only"
    echo "  $0 ep2      # Run profiling for EP=2 only"
    echo "  $0 both     # Run both (default)"
    exit 1
fi

run_profiling() {
    local ep=$1
    local config_file="$SCRIPT_DIR/profile_${ep}_config.toml"
    local log_file="${ep}_profile_run.log"

    echo "=========================================="
    echo "Running profiling for $ep..."
    echo "=========================================="
    echo ""
    echo "Config: $config_file"
    echo "Log: $log_file"
    echo ""

    NGPU=4 \
    CONFIG_FILE="$config_file" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    env/bin/torchrun \
        --nproc_per_node=4 \
        --rdzv_backend c10d \
        --rdzv_endpoint=localhost:0 \
        --local-ranks-filter 0 \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --job.config_file "$config_file" \
        2>&1 | tee "$log_file"

    echo ""
    echo "$ep profiling complete!"
    echo ""
}

if [ "$EP_DEGREE" = "both" ] || [ "$EP_DEGREE" = "ep2" ]; then
    run_profiling "ep2"
fi

if [ "$EP_DEGREE" = "both" ] || [ "$EP_DEGREE" = "ep1" ]; then
    run_profiling "ep1"
fi

echo "=========================================="
echo "All profiling runs complete!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "  ./scripts/ep/compare_profiles.sh"
