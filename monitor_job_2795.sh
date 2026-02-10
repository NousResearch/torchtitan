#!/bin/bash
# Monitor job 2795

echo "Monitoring Job 2795..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "========================================"
    echo "Job 2795 Monitor - $(date)"
    echo "========================================"
    echo ""
    
    # Check job status
    JOB_STATUS=$(squeue -j 2795 --format="%.8T" -h 2>/dev/null)
    
    if [ -z "$JOB_STATUS" ]; then
        echo "Job 2795: COMPLETED or NOT FOUND"
        echo ""
        echo "Checking output files..."
        ls -lh outputs/cp_sweep_local/*.out 2>/dev/null
        break
    else
        echo "Job Status: $JOB_STATUS"
        squeue -j 2795 --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
        echo ""
        
        if [ "$JOB_STATUS" == "RUNNING" ]; then
            echo "Job is RUNNING! Showing last 30 lines of output:"
            echo "----------------------------------------"
            tail -30 outputs/cp_sweep_local/cp16_lbs8_16n_2795.out 2>/dev/null || echo "Output file not yet available"
        fi
    fi
    
    sleep 30
done
