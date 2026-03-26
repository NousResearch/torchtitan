#!/bin/bash
export CPATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))"):$CPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_BUFFSIZE=2097152

python3 submit_sft.py \
  --config_file torchtitan/models/qwen3/train_configs/qwen3_next_30b_a3b_base_sft_combined_100_100.toml \
  --hf_token YOUR_HF_TOKEN_HERE \
  --partition all \
  --cpus-per-task 156 \
  --conda None \
  --venv /home/tinuviel/torchtitan/.venv \
  --n_nodes 2 \
  --job_name qwen3-base-sft-100-50 \
  --slurm_exclude gpu-dp-rgdrx-pmszc,gpu-dp-rgdrx-5vjwk \
  --wandb_team nous_research \
  --wandb_project science_qa \
  --wandb_run_name "qwen3-30b-a3b-base-sft-combined-ds-100-50"
