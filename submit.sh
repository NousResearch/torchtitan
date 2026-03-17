#!/bin/bash
python submit_sft.py \
  --config_file torchtitan/models/qwen3/train_configs/qwen3_30b_a3b_thinking_sft.toml \
  --partition all \
  --cpus-per-task 156 \
  --n_nodes 1 \
  --job_name qwen3-sft-flex \
  --wandb_team nous_research \
  --wandb_project science_qa \
  --wandb_run_name "qwen3-30b-a3b-thinking-sft-science_qa_20260128_unverified"