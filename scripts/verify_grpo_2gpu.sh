#!/usr/bin/bash
# 2-GPU Verification Script for GRPO Tier 1 Infrastructure

set -ex

NGPU=2
CONFIG_FILE="./torchtitan/models/llama3/train_configs/verify_grpo_2gpu.toml"

# Run with torchrun - this assumes a 2-GPU local setup (e.g. 2x 3090)
# We use log-rank 0 filter to keep output clean, but keep both for debugging
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--role rank --tee 3 \
-m torchtitan.grpo_train --job.config_file ${CONFIG_FILE} "$@"
