#!/bin/bash
printenv
ulimit -n 32000
# if NODEID == 0...
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    # Kill any stale run-api or env server processes from previous jobs
    pkill -f 'run-api' || true
    fuser -k 8001/tcp || true
    sleep 1
    # Create the trajectory handler stuff
    echo "Starting job at $(date)"
    # Start trajectory handler
    echo "Starting trajectory handler..."
    ${API_ENV}/bin/run-api --port 8001 > ${LOGDIR}/api.log 2>&1 &
    ${API_ENV}/bin/python $PYTHON_SCRIPT serve --slurm=False $PYTHON_ARGS > ${LOGDIR}/env_server.log 2>&1 &
    echo "Started trajectory handler..."
fi
echo $SLURM_NODEID ", " $NUM_TRAINING_NODES
# now, if we're within the number of nodes allocated to training...
if [[ "$SLURM_NODEID" -lt "$NUM_TRAINING_NODES" ]]; then
    cd $TRAIN_PATH
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}

    echo Node IP: $head_node_ip
    export LOGLEVEL=INFO
    # Enable for A100
#    export FI_PROVIDER="efa"
    # Ensure that P2P is available
    # export NCCL_P2P_DISABLE=1
#    export NCCL_IB_DISABLE=1

    # debugging flags (optional)
    export NCCL_DEBUG=WARN
    export PYTHONFAULTHANDLER=1
    # optional debug settings
    # export NCCL_DEBUG=INFO
    # NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

#    export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    export CUDA_LAUNCH_BLOCKING=0
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

    # on your cluster you might need these:
    # set the network interface
#    export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
#    export NCCL_BUFFSIZE=2097152
#    export TORCH_DIST_INIT_BARRIER=1
#    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

#    dcgmi profile --pause
    # adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
    # to your specific node count, and update target launch file.
    ${TRAIN_ENV}/bin/torchrun --nproc_per_node 8 --nnodes ${NUM_TRAINING_NODES} --rdzv_id 101 --rdzv_backend static --rdzv_endpoint="$head_node_ip:29500" --node-rank=$SLURM_NODEID --role rank --tee 3 \
-m torchtitan.grpo_train --job.config_file ${CONFIG_FILE}  --grpo.sglang_slurm_num_nodes ${NUM_INFERENCE_NODES} ${TRAINING_ARGS}
    scancel $SLURM_JOBID
#    dcgmi profile --resume
# else we're inferencing...
else

    # Kill any stale vllm processes from previous jobs on this inference node
    pkill -f 'vllm_runner' || true
    pkill -f 'vllm.entrypoints' || true
    sleep 3

    # GPUs 0+1 are reserved for the judge (tp=2); rollout workers use GPUs 2-7
    export NUM_ROLLOUT_GPUS_PER_NODE=6
    export MIN_ROLLOUT_GPU=2

    # Setup 6 rollout vllm instances on GPUs 2-7
    echo "Starting vllm instances..."

    export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_${SLURM_NODEID}
    mkdir -p ${TRITON_CACHE_DIR}
    export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}_${SLURM_NODEID}
    mkdir -p ${FLASHINFER_WORKSPACE_BASE}

    # Startup wandb monitoring...
    API_ADDR="http://${head_node_ip}:8000"
    ${API_ENV}/bin/inference-node-wandb-watcher --api_addr ${API_ADDR} --tp 1 --node_num ${SLURM_NODEID} > ${LOGDIR}/wandb_${SLURM_NODEID}.log 2>&1  &


    # Start judge model (Qwen3.5-27B, tp=2) on GPUs 0+1, port 9010
    JUDGE_MODEL=${JUDGE_MODEL:-"Qwen/Qwen3.5-27B"}
    JUDGE_PORT=${JUDGE_PORT:-9010}
    echo "Starting judge vllm on GPUs 0+1, port ${JUDGE_PORT}, model ${JUDGE_MODEL}"
    JUDGE_ENV=${JUDGE_ENV:-"/home/mormio/miniconda3/envs/grpo-vllm-judge"}
    LD_PRELOAD="${JUDGE_ENV}/lib/libstdc++.so.6" FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 nohup ${JUDGE_ENV}/bin/vllm serve ${JUDGE_MODEL} \
      --host 0.0.0.0 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 65536 \
      --dtype="bfloat16" \
      --reasoning-parser qwen3 \
      --language-model-only \
      --port ${JUDGE_PORT} > ${LOGDIR}/vllm_judge.log 2>&1 &
    sleep 5

    PORT_BASE=9000

    # Start 8 vllm instances on GPUs 0-3
    # this assumes you can run it with tp=1
    # if not, well, good luck with single node training, I'll pray for you
    LOG_OFFSET=$((SLURM_NODEID * 8))
    for i in {2..6}; do
        GPU_ID=$i
        LOG_ID=$((GPU_ID + LOG_OFFSET))
        PORT=$((PORT_BASE + i))
        echo "Starting vllm instance on GPU $GPU_ID, logdir $LOG_ID, port $PORT"
        CUDA_VISIBLE_DEVICES=$GPU_ID nohup ${VLLM_ENV}/bin/python -m torchtitan.grpo.vllm_handling.vllm_runner \
          --model $MODEL_NAME \
          --host 0.0.0.0 \
          --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.80} \
          --max-model-len ${VLLM_MAX_MODEL_LEN:-65536} \
          --dtype="bfloat16" \
          --log-level="error" \
          --port $PORT > ${LOGDIR}/vllm_${LOG_ID}.log 2>&1 &
        sleep 3  # wait so vllm can find ports without conflicts :)
    done
    GPU_ID=7
    LOG_ID=$((GPU_ID + LOG_OFFSET))
    PORT=$((PORT_BASE + 7))
    echo "Starting vllm instance on GPU 7, port 9007"
    CUDA_VISIBLE_DEVICES=7 nohup ${VLLM_ENV}/bin/python -m torchtitan.grpo.vllm_handling.vllm_runner \
      --model $MODEL_NAME \
      --host 0.0.0.0 \
      --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.80} \
      --max-model-len ${VLLM_MAX_MODEL_LEN:-65536} \
      --dtype="bfloat16" \
      --log-level="error" \
      --port 9007 > ${LOGDIR}/vllm_${LOG_ID}.log 2>&1 &
    wait
fi
