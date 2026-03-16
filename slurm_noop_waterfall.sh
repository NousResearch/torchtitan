#!/bin/bash
#SBATCH --job-name=noop_lvlX
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --time=00:30:00
#SBATCH --output=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/noop_lvl_%j.out
#SBATCH --error=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/noop_lvl_%j.err

# Waterfall no-op experiment. Set NOOP_LEVEL env var:
# 0 = baseline, 1 = no linear, 2 = +no attn, 3 = +no MoE

NOOP_LEVEL=${NOOP_LEVEL:-0}

TORCHTITAN_DIR="/home/phuc/workspace/moe/small_prs/pr011_kimi_1t_profiling/torchtitan"
VENV_PATH="${TORCHTITAN_DIR}/.venv"
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/seqlen_8k_cp1_lbs2_deepep.toml"

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
NNODES=${#NODES[@]}

echo "Job ID: $SLURM_JOB_ID"
echo "NOOP_LEVEL=${NOOP_LEVEL}"
echo "  0=baseline, 1=no linear, 2=+no attn, 3=+no MoE"
echo "Nodes ($NNODES): ${NODES[@]}"

TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} -m torchtitan.train --job.config_file ${CONFIG_FILE}"

PIDS=()
for node in "${NODES[@]}"; do
    ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && export LOGLEVEL=INFO && export FI_PROVIDER=efa && export NCCL_DEBUG=WARN && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib/:\$LD_LIBRARY_PATH && export NCCL_SOCKET_IFNAME=bond0 && export NCCL_BUFFSIZE=2097152 && export TORCH_DIST_INIT_BARRIER=1 && export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 && export HF_HOME=/tmp/hf_cache && export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15 && export NOOP_LEVEL=${NOOP_LEVEL} && ${TORCHRUN_CMD}" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
