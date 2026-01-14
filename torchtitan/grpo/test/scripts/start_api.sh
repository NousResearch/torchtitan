#!/bin/bash
# Start Atropos API Server

set -e

echo "========================================"
echo "Starting Atropos API Server"
echo "========================================"

source /home/nightwing/Projects/torchtitan/.venv/bin/activate

# Change to Atropos directory
cd /home/shared/atropos

# Start the API server
# The server will listen on http://localhost:8000
echo "Starting API server on http://localhost:8000"
run-api
