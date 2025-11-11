#!/bin/bash
# Convenient script to compare EP profiling results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "=========================================="
echo "EP Performance Analysis Tool"
echo "=========================================="
echo ""

# Check if trace files exist
if [ ! -f "./outputs_profile_ep2/profile_trace/iteration_5/rank0_trace.json" ]; then
    echo "ERROR: EP=2 trace file not found!"
    echo "Please run profiling first with:"
    echo "  ./scripts/ep/run_profiling.sh ep2"
    exit 1
fi

if [ ! -f "./outputs_profile_ep1/profile_trace/iteration_5/rank0_trace.json" ]; then
    echo "ERROR: EP=1 trace file not found!"
    echo "Please run profiling first with:"
    echo "  ./scripts/ep/run_profiling.sh ep1"
    exit 1
fi

echo "Running detailed trace analysis..."
echo ""

env/bin/python "$SCRIPT_DIR/detailed_trace_analysis.py"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "For more details, see:"
echo "  - scripts/ep/EP_PERFORMANCE_ANALYSIS.md (comprehensive report)"
echo "  - ep1_profile_run.log (EP=1 training log)"
echo "  - ep2_profile_run.log (EP=2 training log)"
