# flake8: noqa: F401
# NOTE: The old monkeypatch approach (replacing GPUModelRunner) required fork
# mode, which breaks on setups where CUDA gets initialized before the fork.
# Weight updater spawning is now handled via --worker-cls pointing to
# torchtitan.grpo.vllm_handling.vllm_patching.patched_worker.PatchedWorker

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # noqa: F401
from vllm.v1.engine.async_llm import AsyncLLM

AsyncLLMEngine = AsyncLLM
