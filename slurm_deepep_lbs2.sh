#!/bin/bash
#SBATCH --job-name=kimi_deepep
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --time=00:30:00
#SBATCH --output=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/deepep_lbs2_%j.out
#SBATCH --error=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/deepep_lbs2_%j.err

# DeepEP test: same as 726 TPS baseline but using DeepEP instead of NCCL all-to-all

TORCHTITAN_DIR="/home/phuc/workspace/moe/small_prs/pr011_kimi_1t_profiling/torchtitan"
VENV_PATH="${TORCHTITAN_DIR}/.venv"
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/seqlen_8k_cp1_lbs2_deepep.toml"

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
NNODES=${#NODES[@]}

echo "Job ID: $SLURM_JOB_ID"
echo "DeepEP test: LBS=2, seq=8k, CP=1, EP=64 — comparing with 726 TPS NCCL baseline"
echo "Nodes ($NNODES): ${NODES[@]}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} -m torchtitan.train --job.config_file ${CONFIG_FILE}"

PIDS=()
for node in "${NODES[@]}"; do
    ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && export LOGLEVEL=INFO && export FI_PROVIDER=efa && export NCCL_DEBUG=WARN && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib/:\$LD_LIBRARY_PATH && export NCCL_SOCKET_IFNAME=bond0 && export NCCL_BUFFSIZE=2097152 && export TORCH_DIST_INIT_BARRIER=1 && export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 && export HF_HOME=/tmp/hf_cache && export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15 && ${TORCHRUN_CMD}" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
