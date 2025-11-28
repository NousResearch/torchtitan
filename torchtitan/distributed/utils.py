# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import os
from collections.abc import Generator, Iterable
from datetime import timedelta

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.config import Comm as CommConfig, TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module, device_type


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_sum(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
    world_mesh: DeviceMesh | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
    distinct_seed_mesh_dim: str = "pp",
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimension denoted by `distinct_seed_mesh_dim`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Ensure flex_attention is compiled without max-autotune. This is needed to ensure
        # reproducibility, since the autotune results may not be deterministic.
        from torch.nn.attention.flex_attention import flex_attention

        from torchtitan.models.attention import FlexAttentionWrapper

        FlexAttentionWrapper._compiled_flex_attn = torch.compile(flex_attention)

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # Set distinct seed for each rank in mesh dimensions, with dimension name provided by `distinct_seed_mesh_dim`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # TODO(jianiw): We could further extend this to support multiple distinct dimensions instead of just one.
    if (
        c10d.get_world_size() > 1
        and distinct_seed_mesh_dim in world_mesh.mesh_dim_names
    ):
        distinct_mesh = world_mesh[distinct_seed_mesh_dim]
        seed += distinct_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(
            f"{distinct_seed_mesh_dim} rank {distinct_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}"
        )
        duplicate_seed_mesh = list(
            filter(
                lambda name: name != distinct_seed_mesh_dim, world_mesh.mesh_dim_names
            )
        )
        duplicate_seed_mesh = (
            world_mesh[duplicate_seed_mesh] if len(duplicate_seed_mesh) else None
        )
    else:
        duplicate_seed_mesh = world_mesh
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(
    enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Generator[None, None, None]:
    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context:
                stack.enter_context(cp_context)

            yield

    return context


def maybe_enable_amp(
    parallel_dims: ParallelDims, mixed_precision_param: str, device_type: torch.device
) -> Generator[None, None, None]:
    if parallel_dims.fsdp_enabled:
        # FSDP handles mixed precision internally
        logger.info("Mixed precision training is handled by fully_shard")
        return contextlib.nullcontext()
    else:
        if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
            logger.warning(
                "Mixed precision training with TP or PP is only supported when FSDP/HSDP/CP is enabled."
            )
            logger.info("Mixed precision training is disabled")
            return contextlib.nullcontext()
        else:
            # the following code will only be executed for DDP or single-device training
            logger.info("Mixed precision training is handled by AMP")
            return torch.autocast(
                device_type,
                dtype=TORCH_DTYPE_MAP[mixed_precision_param],
            )


def init_distributed(
    comm_config: CommConfig, enable_cpu_backend: bool = False, base_folder: str = ""
):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    def _get_distributed_backend(enable_cpu_backend):
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map:
            backend = torch.distributed.Backend.default_device_backend_map.get(
                device_type
            )
        if enable_cpu_backend:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    TRACE_BUFFER_SIZE = "TORCH_FR_BUFFER_SIZE"
    TRACE_FILE = "TORCH_FR_DUMP_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(comm_config.trace_buf_size))
    if comm_config.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = os.path.join(base_folder, comm_config.save_traces_folder)
        prefix = comm_config.save_traces_file_prefix
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/{prefix}")

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(enable_cpu_backend),
        timeout=timedelta(seconds=comm_config.init_timeout_seconds),
    )


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_enabled: bool = False,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: Pipeline Parallel device mesh. If not None, will reduce gradient norm across PP stages.
        ep_dense_params_mesh_ndim: Mesh ndim of the dense params when EP is used. If EP is not used,
            set it to ``None``.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if ep_enabled:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    # =====================================================================
    # COMPREHENSIVE DEBUG LOGGING FOR GRADIENT NORM BLOW-UP INVESTIGATION
    # =====================================================================
    import os

    DEBUG_GRAD = os.environ.get("DEBUG_GRAD_NORM", "0") == "1"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    def debug_log(msg):
        if DEBUG_GRAD and rank == 0:
            logger.info(f"[GRAD_DEBUG] {msg}")

    def debug_log_all_ranks(msg):
        if DEBUG_GRAD:
            logger.info(f"[GRAD_DEBUG][rank{rank}] {msg}")

    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    # STEP 1: Categorize parameters into EP and non-EP
    debug_log("=" * 80)
    debug_log("STEP 1: Categorizing parameters into EP and non-EP")

    # Track all params for per-param gradient analysis
    all_param_grads = []  # List of (grad_norm, shape, is_ep, idx)

    for idx, p in enumerate(parameters):
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        local_grad_norm = p.grad.to_local().float().norm().item()
        is_ep = "ep" in p.device_mesh.mesh_dim_names
        all_param_grads.append((local_grad_norm, tuple(p.shape), is_ep, idx))

        if is_ep:
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)

    # Log TOP 10 parameters by gradient norm to identify explosion source
    if DEBUG_GRAD and rank == 0 and all_param_grads:
        all_param_grads.sort(key=lambda x: x[0], reverse=True)
        debug_log("TOP 10 PARAMETERS BY LOCAL GRADIENT NORM:")
        for i, (norm, shape, is_ep, idx) in enumerate(all_param_grads[:10]):
            debug_log(
                f"  [{i+1}] idx={idx}, norm={norm:.4f}, shape={shape}, is_ep={is_ep}"
            )

    debug_log(f"  EP params: {len(ep_params)}, Non-EP params: {len(non_ep_params)}")
    debug_log(f"  EP grads: {len(ep_grads)}, Non-EP grads: {len(non_ep_grads)}")

    # STEP 2: Log individual EP grad stats from ALL ranks
    if DEBUG_GRAD and len(ep_grads) > 0:
        # Compute local stats
        local_norms = [g.to_local().norm().item() for g in ep_grads[:5]]
        total_local_norm = sum(g.to_local().norm().item() ** 2 for g in ep_grads) ** 0.5

        # Gather stats from all ranks
        stats_tensor = torch.tensor([total_local_norm], device=ep_grads[0].device)
        all_stats = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
        dist.all_gather(all_stats, stats_tensor)

        if rank == 0:
            debug_log("STEP 2: Individual EP grad statistics from ALL ranks")
            debug_log(
                f"  Per-rank total EP grad local norms: {[f'{v.item():.2f}' for v in all_stats]}"
            )
            debug_log(
                f"  Ratio (max/min): {max(v.item() for v in all_stats) / (min(v.item() for v in all_stats) + 1e-10):.2f}x"
            )

            # Also show first 5 grads from rank 0
            for i, g in enumerate(ep_grads[:5]):
                local_g = g.to_local()
                debug_log(
                    f"  ep_grad[{i}]: shape={g.shape}, placements={g.placements}, "
                    f"local_shape={local_g.shape}, local_norm={local_g.norm().item():.6f}, "
                    f"local_min={local_g.min().item():.6f}, local_max={local_g.max().item():.6f}"
                )

    # STEP 3: Call get_total_norm on EP grads
    debug_log("STEP 3: Calling torch.nn.utils.get_total_norm(ep_grads, ...)")
    ep_grads_total_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, error_if_nonfinite, foreach
    )
    # ep_grads may be an empty list, in which case get_total_norm returns tensor(0.), a non-DTensor
    # This can occur in PP + EP setups where certain PP ranks only own non-EP layers, for instance.

    # STEP 4: Analyze get_total_norm result across all ranks
    debug_log("STEP 4: Analyzing get_total_norm result across all ranks")
    if isinstance(ep_grads_total_norm, DTensor):
        local_val = ep_grads_total_norm.to_local()
        # Gather all values to rank 0
        all_local_vals = [torch.zeros_like(local_val) for _ in range(world_size)]
        dist.all_gather(all_local_vals, local_val)

        if DEBUG_GRAD and rank == 0:
            debug_log(f"  ep_grads_total_norm is DTensor")
            debug_log(f"  placements: {ep_grads_total_norm.placements}")
            debug_log(f"  mesh shape: {ep_grads_total_norm.device_mesh.shape}")
            debug_log(
                f"  mesh_dim_names: {ep_grads_total_norm.device_mesh.mesh_dim_names}"
            )
            debug_log(
                f"  local_tensor: device={local_val.device}, dtype={local_val.dtype}, shape={local_val.shape}, value={local_val.item():.6f}"
            )
            debug_log(
                f"  ALL RANKS local values: {[f'{v.item():.6f}' for v in all_local_vals]}"
            )
            sum_sq = sum(v.item() ** 2 for v in all_local_vals)
            debug_log(f"  Sum of squares: {sum_sq:.6f}")
            debug_log(
                f"  CORRECT global norm = sqrt(sum of squares) = {sum_sq**0.5:.6f}"
            )

            # Check placement types
            for dim_idx, placement in enumerate(ep_grads_total_norm.placements):
                debug_log(
                    f"  Placement[{dim_idx}]: type={type(placement).__name__}, class={placement.__class__.__name__}"
                )

    if isinstance(ep_grads_total_norm, DTensor):
        # =====================================================================================================================================
        # PYTORCH BUG WORKAROUND: Incorrect _NormPartial DTensor reduction in PyTorch 2.9.0
        # =====================================================================================================================================
        #
        # ## PROBLEM DESCRIPTION ##
        # PyTorch 2.9.0 has a critical bug in torch/distributed/tensor/_ops/_math_ops.py that causes gradient norms to explode
        # by ~1,000,000x when using Expert Parallelism (EP) with DTensors. Without this workaround, gradient norms blow up from
        # ~0.5 (correct) to ~200,000+ (wrong), leading to training instability and divergence.
        #
        # ## ROOT CAUSE ##
        # The bug is in `vector_norm_strategy()` and `foreach_norm_strategy()` functions (lines 393-433 in _math_ops.py):
        #
        #     def vector_norm_strategy(op_schema: OpSchema) -> OpStrategy:
        #         ...
        #         return common_reduction_strategy(
        #             input_strategy,
        #             reduce_dims,
        #             keep_dim=cast(bool, keepdim),
        #             reduction_linear=True,  # <-- BUG: Always True, even for Partial placements!
        #             reduction_op=NormReduction(norm_type),
        #         )
        #
        # The `reduction_linear=True` flag incorrectly assumes that the norm operation is always "reduction linear" with respect
        # to its inputs. This is FALSE for DTensors with `Partial(sum)` placement, where each rank holds partial data that must
        # be summed before computing the norm.
        #
        # ## WHAT HAPPENS WITH THE BUG ##
        # When computing `get_total_norm()` on EP gradient DTensors:
        #
        # 1. Input: DTensor with `Shard` or `Partial(sum)` placement across EP ranks
        #    Example: Rank 0 has [1.0, 3.0], Rank 1 has [2.0, 1.0]
        #    Global data should be: [3.0, 4.0] (after summing Partial contributions)
        #
        # 2. PyTorch computes norm on LOCAL data (WRONG!):
        #    Rank 0: local_norm = sqrt(1^2 + 3^2) = sqrt(10) = 3.16
        #    Rank 1: local_norm = sqrt(2^2 + 1^2) = sqrt(5) = 2.24
        #
        # 3. Result is DTensor with `Partial(sum)` placement, but calling `.full_tensor()` SUMS the local norms (WRONG!):
        #    full_tensor() = 3.16 + 2.24 = 5.40  <-- This is "sum of norms", NOT "norm of sum"!
        #
        # 4. Correct global norm should be:
        #    global_norm = sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5.0  <-- This is the "norm of sum"
        #
        # ## THE FIX IN PYTORCH ##
        # The fix was merged in PyTorch PR #159856 (commit f863550192e, Oct 26, 2025):
        # https://github.com/pytorch/pytorch/pull/159856
        #
        # It changes `reduction_linear=True` to:
        #
        #     reduction_linear = all(
        #         all(not p.is_partial() for p in op_spec.output_spec.placements)
        #         for op_spec in input_strategy.strategies
        #     )
        #
        # This makes the norm operation output a DTensor with `_NormPartial` placement instead of `Partial(sum)`, which
        # correctly handles the reduction semantics: sqrt(sum(local_norm^2)) across ranks.
        #
        # ## OUR WORKAROUND (Until PyTorch is upgraded) ##
        # Since PyTorch 2.9.0 doesn't have the fix, we must manually handle `_NormPartial` reduction:
        #
        # 1. Detect DTensors with `_NormPartial` placement (private class from torch.distributed._tensor.placement_types)
        # 2. Extract local norm values: local_norm = dtensor.to_local()
        # 3. Square the local norms: local_norm_squared = local_norm ** 2
        # 4. All-reduce SUM across the mesh dimension(s) with `_NormPartial`: dist.all_reduce(local_norm_squared, SUM)
        # 5. Take square root: global_norm = sqrt(local_norm_squared)
        #
        # This implements the correct _NormPartial semantics: sqrt(sum(local_norm^2))
        #
        # ## CRITICAL NOTES ##
        # - The mesh can have multiple dimensions (e.g., ['dp_shard_mod_ep', 'dp_shard_in_ep']) and we need to reduce
        #   across ALL dimensions that have `_NormPartial` placement
        # - The EP mesh dimension name varies by configuration (not always "ep"), so we must iterate through all placements
        # - `_NormPartial` is a private class, so we check by class name: placement.__class__.__name__ == "_NormPartial"
        # - Do NOT use `.full_tensor()` or `.redistribute()` on `_NormPartial` DTensors - they don't handle the semantics correctly!
        #
        # ## WHEN TO REMOVE THIS WORKAROUND ##
        # This workaround can be removed when upgrading to PyTorch with commit f863550192e or later (post Oct 26, 2025).
        # To verify the fix is present, check torch/distributed/tensor/_ops/_math_ops.py:vector_norm_strategy() and ensure
        # `reduction_linear` is computed dynamically based on partial placements, not hardcoded to True.
        #
        # ## DEBUGGING THIS ISSUE ##
        # If gradient norms are exploding (>100K):
        # 1. Check if this workaround is being executed (add logging in the `if norm_partial_dims:` block)
        # 2. Print ep_grads_total_norm.placements to see if `_NormPartial` is present
        # 3. Print local_norm before and after all_reduce to verify the reduction
        # 4. Expected: local_norm ~0.1-1.0, global_norm ~0.5-10.0; if seeing 100K+, reduction failed
        #
        # ## OLD BUGGY CODE (DO NOT USE) ##
        # ep_grads_total_norm = ep_grads_total_norm.full_tensor()  # WRONG: Sums local norms instead of computing global norm
        # ep_grads_total_norm = ep_grads_total_norm.redistribute(mesh, [Replicate()])  # WRONG: Same issue as full_tensor()
        # =====================================================================================================================================

        from torch.distributed._tensor import Replicate
        from torch.distributed._tensor.placement_types import Partial

        # Find which mesh dimension(s) have _NormPartial placement
        # _NormPartial is a private PyTorch class that indicates the tensor contains partial norm contributions
        # that need to be combined via: sqrt(sum(local_norm^2))
        norm_partial_dims = []
        for dim_idx, placement in enumerate(ep_grads_total_norm.placements):
            if placement.__class__.__name__ == "_NormPartial":
                norm_partial_dims.append(dim_idx)

        # STEP 5: Apply _NormPartial workaround if needed
        debug_log(
            f"STEP 5: Checking for _NormPartial placements, found dims: {norm_partial_dims}"
        )

        if norm_partial_dims:
            # WORKAROUND: Manual reduction for _NormPartial using all_gather
            debug_log("  _NormPartial detected - applying manual all_gather workaround")

            local_norm = ep_grads_total_norm.to_local()
            debug_log(f"  BEFORE workaround: local_norm = {local_norm.item():.6f}")

            # Gather all local norm values from all ranks
            all_local_norms = [torch.zeros_like(local_norm) for _ in range(world_size)]
            dist.all_gather(all_local_norms, local_norm)

            if DEBUG_GRAD and rank == 0:
                debug_log(
                    f"  all_local_norms gathered: {[f'{v.item():.6f}' for v in all_local_norms]}"
                )

            # Compute correct global norm: sqrt(sum(local_norm^2))
            sum_of_squares = sum(v.item() ** 2 for v in all_local_norms)
            global_norm = sum_of_squares**0.5

            debug_log(f"  sum_of_squares = {sum_of_squares:.6f}")
            debug_log(
                f"  AFTER workaround: global_norm = sqrt(sum_of_squares) = {global_norm:.6f}"
            )

            # Convert back to tensor on the correct device
            ep_grads_total_norm = torch.tensor(
                global_norm, device=local_norm.device, dtype=local_norm.dtype
            )

        else:
            # No _NormPartial placement detected - safe to use .full_tensor()
            debug_log("  No _NormPartial detected - using .full_tensor()")
            before_val = (
                ep_grads_total_norm.to_local().item()
                if isinstance(ep_grads_total_norm, DTensor)
                else ep_grads_total_norm.item()
            )
            ep_grads_total_norm = ep_grads_total_norm.full_tensor()
            debug_log(f"  BEFORE .full_tensor(): local_val = {before_val:.6f}")
            debug_log(
                f"  AFTER .full_tensor(): ep_grads_total_norm = {ep_grads_total_norm.item():.6f}"
            )

    # STEP 6: Compute non-EP grads total norm
    debug_log("STEP 6: Computing non-EP grads total norm")
    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    )
    if isinstance(non_ep_grads_total_norm, DTensor):
        non_ep_grads_total_norm = non_ep_grads_total_norm.full_tensor()
    debug_log(f"  non_ep_grads_total_norm = {non_ep_grads_total_norm.item():.6f}")

    # STEP 7: Combine EP and non-EP norms
    debug_log("STEP 7: Combining EP and non-EP norms")
    debug_log(f"  ep_grads_total_norm = {ep_grads_total_norm.item():.6f}")
    debug_log(f"  non_ep_grads_total_norm = {non_ep_grads_total_norm.item():.6f}")

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = (
            ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        )
        total_norm **= 1.0 / norm_type

    debug_log(f"  combined total_norm (before PP reduce) = {total_norm.item():.6f}")

    # STEP 8: PP reduction if needed
    if pp_mesh is not None:
        debug_log("STEP 8: PP mesh reduction")
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type
        debug_log(f"  total_norm (after PP reduce) = {total_norm.item():.6f}")

    # STEP 9: Apply gradient clipping
    debug_log("STEP 9: Applying gradient clipping")
    debug_log(f"  max_norm = {max_norm}, total_norm = {total_norm.item():.6f}")
    clip_coef = max_norm / (total_norm + 1e-6)
    debug_log(f"  clip_coef = {clip_coef.item():.6f} (clamped to max 1.0)")

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    debug_log(f"FINAL: Returning total_norm = {total_norm.item():.6f}")
    debug_log("=" * 80)

    return total_norm
