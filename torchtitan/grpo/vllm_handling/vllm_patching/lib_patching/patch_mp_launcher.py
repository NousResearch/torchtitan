# flake8: noqa: F401
import vllm.v1.worker.gpu_model_runner
import vllm.v1.worker.gpu_worker
from torchtitan.grpo.vllm_handling.vllm_patching.lib_patching.gpu_model_runner import (
    PatchedGPUModelRunner,
)

# Patch GPU Runner in gpu_model_runner (where Worker imports it from dynamically)
vllm.v1.worker.gpu_model_runner.GPUModelRunner = PatchedGPUModelRunner
# Also patch gpu_worker for any direct references
vllm.v1.worker.gpu_worker.GPUModelRunner = PatchedGPUModelRunner

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.gpu_worker import Worker

# Patch MP launcher with this context...
from vllm.v1.worker.worker_base import WorkerWrapperBase

AsyncLLMEngine = AsyncLLM
