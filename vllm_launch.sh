#!/bin/bash
printenv
ulimit -n 32000
echo "Cleaning up GPU processes..."
sudo pkill -f torchrun 2>/dev/null
sudo pkill -f "torchtitan.train" 2>/dev/null
sleep 1
# Force kill anything still on the GPUs
sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null

export HF_HOME=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_datasets
export TRITON_CACHE_DIR=/scratch/$USER/triton_cache

mkdir -p /scratch/$USER/hf_cache
mkdir -p /scratch/$USER/hf_datasets
mkdir -p /scratch/$USER/triton_cache

# if NODEID == 0...
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    # Create the trajectory handler stuff
    echo "Starting job at $(date)"
    source ${API_ENV}/bin/activate
    # Start trajectory handler
    echo "Starting trajectory handler..."
    run-api > ${LOGDIR}/api.log 2>&1 &
    python $PYTHON_SCRIPT serve --slurm=True $PYTHON_ARGS > ${LOGDIR}/env_server.log 2>&1 &
    deactivate
    echo "Started trajectory handler..."
fi
echo $SLURM_NODEID ", " $NUM_TRAINING_NODES
# now, if we're within the number of nodes allocated to training...
if [[ "$SLURM_NODEID" -lt "$NUM_TRAINING_NODES" ]]; then
    source ${TRAIN_ENV}/bin/activate
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
    torchrun --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint="$head_node_ip:29500"  --role rank --tee 3 --nnodes ${NUM_TRAINING_NODES} \
-m torchtitan.grpo_train --job.config_file ${CONFIG_FILE}  --grpo.sglang_slurm_num_nodes ${NUM_INFERENCE_NODES} ${TRAINING_ARGS}
    scancel $SLURM_JOBID
#    dcgmi profile --resume
# else we're inferencing...
else

    # Setup vllm instances with model in vllm venv
    echo "Starting vllm instances..."

    VLLM_PP=${VLLM_PP:-1}
    VLLM_DP=${VLLM_DP:-1}
    USE_EP=${USE_EP:-0}
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    GPUS_PER_INSTANCE=$((VLLM_PP * VLLM_DP))
    NUM_INSTANCES=$((8 / GPUS_PER_INSTANCE))

    # Build parallelism args
    PP_ARGS=""
    if [[ "$VLLM_PP" -gt 1 ]]; then
        PP_ARGS="--pipeline-parallel-size $VLLM_PP"
    fi
    DP_ARGS=""
    if [[ "$VLLM_DP" -gt 1 ]]; then
        DP_ARGS="--data-parallel-size $VLLM_DP"
    fi
    EP_ARGS=""
    if [[ "$USE_EP" -eq 1 ]]; then
        EP_ARGS="--enable-expert-parallel"
    fi

    # Startup wandb monitoring...
    source ${API_ENV}/bin/activate
    API_ADDR="http://${head_node_ip}:8000"
    inference-node-wandb-watcher --api_addr ${API_ADDR} --tp 1 --node_num ${SLURM_NODEID} > ${LOGDIR}/wandb_${SLURM_NODEID}.log 2>&1  &

    source ${VLLM_ENV}/bin/activate

    PORT_BASE=9000

    LOG_OFFSET=$((SLURM_NODEID * NUM_INSTANCES))

    # Start all but the last instance in background
    for ((i=0; i<NUM_INSTANCES-1; i++)); do
        START_GPU=$((i * GPUS_PER_INSTANCE))
        GPU_IDS=$(seq -s, $START_GPU $((START_GPU + GPUS_PER_INSTANCE - 1)))
        LOG_ID=$((i + LOG_OFFSET))
        PORT=$((PORT_BASE + i))
        echo "Starting vllm instance $i on GPUs $GPU_IDS, port $PORT"
        CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python -m torchtitan.grpo.vllm_handling.vllm_runner \
          --model $MODEL_NAME \
          --trust-remote-code \
          --host 0.0.0.0 \
          --gpu-memory-utilization 0.75 \
          --dtype="bfloat16" \
          --log-level="error" \
          $PP_ARGS $DP_ARGS $EP_ARGS \
          --port $PORT > ${LOGDIR}/vllm_${LOG_ID}.log 2>&1 &
        sleep 3
    done

    # Last instance in foreground so srun doesn't exit
    i=$((NUM_INSTANCES - 1))
    START_GPU=$((i * GPUS_PER_INSTANCE))
    GPU_IDS=$(seq -s, $START_GPU $((START_GPU + GPUS_PER_INSTANCE - 1)))
    LOG_ID=$((i + LOG_OFFSET))
    PORT=$((PORT_BASE + i))
    echo "Starting vllm instance $i on GPUs $GPU_IDS, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torchtitan.grpo.vllm_handling.vllm_runner \
      --model $MODEL_NAME \
      --trust-remote-code \
      --host 0.0.0.0 \
      --gpu-memory-utilization 0.75 \
      --dtype="bfloat16" \
      --log-level="error" \
      $PP_ARGS $DP_ARGS $EP_ARGS \
      --port $PORT > ${LOGDIR}/vllm_${LOG_ID}.log 2>&1
fi
