#!/bin/bash
python submit_sft.py \
  --config_file torchtitan/models/qwen3/train_configs/qwen3_next_80b_a3b_sft.toml \
  --n_nodes 4 \
  --job_name qwen3-sft-flex \
  --wandb_team nous_research \
  --wandb_project science_qa \
  --wandb_run_name "qwen3-next-80b-a3b-sft-science_qa_20260128_unverified"
