#!/bin/bash
# Start Atropos API Server

set -e

echo "========================================"
echo "Starting Atropos API Server"
echo "========================================"

# Add Atropos to PYTHONPATH
export PYTHONPATH=/home/shared/atropos:$PYTHONPATH

# Change to Atropos directory
cd /home/shared/atropos

# Start the API server
# The server will listen on http://localhost:8000
echo "Starting API server on http://localhost:8000"
run-api
