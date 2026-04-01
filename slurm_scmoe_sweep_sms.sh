#!/bin/bash
#SBATCH --job-name=sms_sweep
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --qos=low
#SBATCH --time=01:00:00
#SBATCH --output=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/sms_sweep_%j.out
#SBATCH --error=/home/phuc/workspace/moe/moe_throughputs/slurm_logs/sms_sweep_%j.err

TORCHTITAN_DIR="/home/phuc/workspace/moe/small_prs/pr_012_shortcut_connected_moe_for_kimi/torchtitan"
VENV_PATH="/home/phuc/workspace/moe/small_prs/pr_012_shortcut_connected_moe_for_kimi/.venv"
CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/seqlen_8k_cp1_lbs2_deepep_scmoe.toml"

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
NNODES=${#NODES[@]}

echo "Job ID: $SLURM_JOB_ID"
echo "ScMoE num_sms sweep: testing 4,6,8,10,12,16,20,30,40"
echo "Nodes ($NNODES): ${NODES[@]}"

run_one() {
    local NUM_SMS=$1
    local PORT=$2
    echo ""
    echo "=========================================="
    echo "  DEEPEP_NUM_SMS=${NUM_SMS} (port ${PORT})"
    echo "=========================================="

    TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID}_${NUM_SMS} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${PORT} -m torchtitan.train --job.config_file ${CONFIG_FILE}"

    PIDS=()
    for node in "${NODES[@]}"; do
        ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && export DEEPEP_NUM_SMS=${NUM_SMS} && export LOGLEVEL=INFO && export FI_PROVIDER=efa && export NCCL_DEBUG=WARN && export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib/:\$LD_LIBRARY_PATH && export NCCL_SOCKET_IFNAME=bond0 && export NCCL_BUFFSIZE=2097152 && export TORCH_DIST_INIT_BARRIER=1 && export FI_EFA_SET_CUDA_SYNC_MEMOPS=0 && export HF_HOME=/tmp/hf_cache && export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15 && ${TORCHRUN_CMD}" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
}

# Run sweep sequentially (each uses 8 nodes, can't run in parallel)
for SMS in 4 6 8 10 12 16 20 30 40; do
    PORT=$((29500 + SMS))
    run_one $SMS $PORT
done
