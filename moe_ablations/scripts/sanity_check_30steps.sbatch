#!/bin/bash
#SBATCH --job-name=sanity_30
#SBATCH --output=/home/phuc/workspace/moe/moe_ablations/torchtitan/moe_ablations/logs/sanity_check_30steps.out
#SBATCH --error=/home/phuc/workspace/moe/moe_ablations/torchtitan/moe_ablations/logs/sanity_check_30steps.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

cd /home/phuc/workspace/moe/moe_ablations/torchtitan
PYTORCH_ALLOC_CONF="expandable_segments:True" /home/phuc/workspace/moe/moe_ablations/torchtitan/env/bin/torchrun --nproc_per_node=8 -m torchtitan.train --job.config-file ./moe_ablations/configs/sanity_check_30steps.toml
