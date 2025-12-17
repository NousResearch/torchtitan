#!/bin/bash
#SBATCH --job-name=tune_v2
#SBATCH --output=/home/phuc/workspace/moe/moe_ablations/torchtitan/tune_v2.out
#SBATCH --error=/home/phuc/workspace/moe/moe_ablations/torchtitan/tune_v2.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

cd /home/phuc/workspace/moe/moe_ablations/torchtitan
PYTORCH_ALLOC_CONF="expandable_segments:True" /home/phuc/workspace/moe/moe_ablations/torchtitan/env/bin/torchrun --nproc_per_node=8 -m torchtitan.train --job.config-file ./torchtitan/experiments/qwen3_next/train_configs/tune_v2.toml
