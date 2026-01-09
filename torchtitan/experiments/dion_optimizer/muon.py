# source link:
# https://github.com/microsoft/dion/blob/main/dion/muon.py
# MuonClip extension: https://arxiv.org/abs/2507.20534 (Kimi K2)

import math

from dataclasses import dataclass, field
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach


@dataclass
class AttentionHeadParams:
    """Container for attention head parameters used in QK-Clip.

    For standard multi-head attention:
        - wq: Query projection weights [head_dim, hidden_dim] or combined [num_heads * head_dim, hidden_dim]
        - wk: Key projection weights [head_dim, hidden_dim] or combined [num_heads * head_dim, hidden_dim]

    For MLA (Multi-head Latent Attention) as in DeepSeek v3:
        - wq_c: Query compression weights (head-specific)
        - wk_c: Key compression weights (head-specific)
        - wq_r: Query rotary weights (head-specific)
        - wk_r: Key rotary weights (shared across heads, not clipped)
    """
    wq: Optional[Tensor] = None
    wk: Optional[Tensor] = None
    # MLA-specific components
    wq_c: Optional[Tensor] = None  # Query compression (head-specific)
    wk_c: Optional[Tensor] = None  # Key compression (head-specific)
    wq_r: Optional[Tensor] = None  # Query rotary (head-specific)
    wk_r: Optional[Tensor] = None  # Key rotary (shared, not clipped)
    head_idx: int = 0
    num_heads: int = 1


@dataclass
class QKClipConfig:
    """Configuration for QK-Clip mechanism.

    QK-Clip prevents attention logit explosion by monitoring per-head maximum
    attention logits and rescaling query/key projection weights when they exceed
    a threshold τ.

    From Kimi K2 paper (https://arxiv.org/abs/2507.20534):
    - γ_h = min(1, τ / S_max^h) where S_max^h is max attention logit for head h
    - Standard attention: Q and K weights scaled by sqrt(γ_h)
    - MLA: Q_c and K_c scaled by sqrt(γ_h), Q_r scaled by γ_h, K_r unchanged

    MLA mode is auto-detected based on whether MLA parameters (wq_c, wk_c, etc.)
    are registered for each attention layer.
    """
    enabled: bool = False
    tau: float = 100.0  # Threshold for attention logit clipping


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        # QK-Clip parameters (MuonClip from Kimi K2)
        qk_clip_config: Optional[QKClipConfig] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # QK-Clip configuration (MuonClip)
        self._qk_clip_config = qk_clip_config or QKClipConfig()
        # Registry mapping layer names to attention head parameters
        # Key: layer_name (str), Value: Dict[head_idx, AttentionHeadParams]
        self._attention_params_registry: Dict[str, Dict[int, AttentionHeadParams]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        # Debug logging for expert verification
        expert_param_count = 0
        muon_expert_count = 0
        adamw_expert_count = 0

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]

            # Check for expert parameters in this group
            for param in group["params"]:
                if hasattr(param, "_param_name"):
                    param_name = param._param_name
                    if self._is_expert_param_name(param_name):
                        expert_param_count += 1
                        if algo == "muon":
                            muon_expert_count += 1
                        elif algo == "adamw":
                            adamw_expert_count += 1

            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Summary logging is handled in parameter classification

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _is_expert_param_name(self, name: str) -> bool:
        """Check if parameter name indicates it's an expert parameter."""
        expert_patterns = [
            "experts.",
            ".expert.",
            "expert_",
            "moe.expert",
            "shared_experts",
            "routed_experts",
            ".experts[",
            ".w1.",
            ".w2.",
            ".w3.",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in expert_patterns)

    # ==================== QK-Clip (MuonClip) Methods ====================

    def register_attention_params(
        self,
        layer_name: str,
        wq: Optional[Tensor] = None,
        wk: Optional[Tensor] = None,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        # MLA-specific parameters
        wq_c: Optional[Tensor] = None,
        wk_c: Optional[Tensor] = None,
        wq_r: Optional[Tensor] = None,
        wk_r: Optional[Tensor] = None,
    ) -> None:
        """Register attention parameters for QK-Clip.

        This method should be called during model initialization to register
        the query and key projection weights for each attention layer.

        Args:
            layer_name: Unique identifier for the attention layer (e.g., "layers.0.attention")
            wq: Query projection weights. Shape [num_heads * head_dim, hidden_dim] or [head_dim, hidden_dim]
            wk: Key projection weights. Shape [num_heads * head_dim, hidden_dim] or [head_dim, hidden_dim]
            num_heads: Number of attention heads
            head_dim: Dimension per head (inferred from wq if not provided)
            wq_c: (MLA) Query compression weights
            wk_c: (MLA) Key compression weights
            wq_r: (MLA) Query rotary weights (head-specific)
            wk_r: (MLA) Key rotary weights (shared, not clipped)
        """
        if not self._qk_clip_config.enabled:
            return

        if layer_name not in self._attention_params_registry:
            self._attention_params_registry[layer_name] = {}

        if head_dim is None and wq is not None:
            # Infer head_dim from weight shape
            head_dim = wq.shape[0] // num_heads

        # Register params for each head
        # MLA params are stored if provided (auto-detected during apply_qk_clip)
        for head_idx in range(num_heads):
            self._attention_params_registry[layer_name][head_idx] = AttentionHeadParams(
                wq=wq,
                wk=wk,
                wq_c=wq_c,
                wk_c=wk_c,
                wq_r=wq_r,
                wk_r=wk_r,
                head_idx=head_idx,
                num_heads=num_heads,
            )

    def apply_qk_clip(
        self,
        attention_max_logits: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Apply QK-Clip to rescale query/key weights based on max attention logits.

        This method should be called AFTER the optimizer step, providing the maximum
        attention logits observed during the forward pass. It rescales the Q/K
        projection weights to prevent attention logit explosion.

        Args:
            attention_max_logits: Dictionary mapping layer names to tensors of max logits
                                  per head. Shape: [num_heads] or scalar for single head.
                                  Example: {"layers.0.attention": tensor([105.2, 98.1, ...])}

        Returns:
            Dictionary mapping layer names to the scaling factors applied (γ_h) for logging.
        """
        if not self._qk_clip_config.enabled:
            return {}

        tau = self._qk_clip_config.tau
        scaling_factors = {}

        for layer_name, max_logits in attention_max_logits.items():
            if layer_name not in self._attention_params_registry:
                continue

            # Ensure max_logits is a tensor
            if not isinstance(max_logits, Tensor):
                max_logits = torch.tensor(max_logits, device=next(iter(self._attention_params_registry[layer_name].values())).wq.device if self._attention_params_registry[layer_name] else "cpu")

            # Handle scalar case (single head)
            if max_logits.ndim == 0:
                max_logits = max_logits.unsqueeze(0)

            layer_scaling = []

            for head_idx, head_params in self._attention_params_registry[layer_name].items():
                if head_idx >= len(max_logits):
                    continue

                s_max_h = max_logits[head_idx].item()

                # Compute scaling factor: γ_h = min(1, τ / S_max^h)
                gamma_h = min(1.0, tau / (s_max_h + 1e-8))
                layer_scaling.append(gamma_h)

                if gamma_h < 1.0:
                    # Need to rescale weights
                    sqrt_gamma = math.sqrt(gamma_h)

                    # Auto-detect MLA mode: if MLA params exist, use MLA-style clipping
                    has_mla_params = head_params.wq_c is not None or head_params.wk_c is not None

                    if has_mla_params:
                        # MLA-specific clipping (Kimi K2 paper):
                        # - Q_c and K_c scaled by sqrt(γ_h)
                        # - Q_r scaled by γ_h
                        # - K_r left unchanged (shared across heads)
                        if head_params.wq_c is not None:
                            _apply_head_scaling(head_params.wq_c, sqrt_gamma, head_idx, head_params.num_heads)
                        if head_params.wk_c is not None:
                            _apply_head_scaling(head_params.wk_c, sqrt_gamma, head_idx, head_params.num_heads)
                        if head_params.wq_r is not None:
                            _apply_head_scaling(head_params.wq_r, gamma_h, head_idx, head_params.num_heads)
                        # wk_r is NOT scaled (shared component)
                    else:
                        # Standard attention: Q and K scaled by sqrt(γ_h)
                        if head_params.wq is not None:
                            _apply_head_scaling(head_params.wq, sqrt_gamma, head_idx, head_params.num_heads)
                        if head_params.wk is not None:
                            _apply_head_scaling(head_params.wk, sqrt_gamma, head_idx, head_params.num_heads)

            scaling_factors[layer_name] = torch.tensor(layer_scaling)

        return scaling_factors

    @property
    def qk_clip_enabled(self) -> bool:
        """Check if QK-Clip is enabled."""
        return self._qk_clip_config.enabled

    @property
    def qk_clip_tau(self) -> float:
        """Get the QK-Clip threshold τ."""
        return self._qk_clip_config.tau

    # ==================== End QK-Clip Methods ====================

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = group["nesterov"]
            flatten = group["flatten"]
            adjust_lr = group["adjust_lr"]

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding dimension
                sharded_mesh_dim = None
                sharded_tensor_dim = None
                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]
                    if len(shard_placements) == 1:
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh"
                        )

                yield AsyncTask(
                    muon_update_batch_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(gradients, self._world_size),
                        M=pad_batch(momentums, self._world_size),
                        lr=lr,
                        momentum=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        nesterov=nesterov,
                        flatten=flatten,
                        adjust_lr=adjust_lr,
                        device_rank=self._device_rank,
                        world_size=self._world_size,
                        shard_dim=sharded_tensor_dim,
                        process_group=self._process_group,
                        newton_schulz_func=self._newton_schulz_func,
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == world_size

    # Expert parameter tracking (logging removed for cleaner output)

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    else:
        # Matrices are not sharded, so we can directly orthogonalize
        # Get a single matrix corresponding to this device
        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        if process_group is not None and process_group.size() > 1:
            # Allocate empty tensors to receive updates from other devices
            U = [torch.empty_like(u) for u in U]

            # All gather orthogonalized results from other devices into buffer
            work = dist.all_gather(
                U, single_matrix.contiguous(), group=process_group, async_op=True
            )
            yield
            work.wait()

        else:
            # Single GPU case, no need to gather
            assert world_size == 1
            U = [single_matrix]

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Lion update.
    """
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


# @torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


# @torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    Always normalizes to 3D before calling newton_schulz_func to avoid torch.compile recompilations.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    # Always ensure 3D input to newton_schulz_func to avoid torch.compile recompilations
    # due to rank mismatch (2D vs 3D tensors triggering separate traces)
    added_batch_dim = False
    if X.ndim == 2:
        X = X.unsqueeze(0)  # Add batch dimension: [M, N] -> [1, M, N]
        added_batch_dim = True

    result = newton_schulz_func(X, epsilon=epsilon)

    if added_batch_dim:
        result = result.squeeze(0)  # Remove batch dimension: [1, M, N] -> [M, N]

    return result.reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[:2]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


# ==================== QK-Clip Helper Functions ====================


@torch.no_grad()
def _apply_head_scaling(
    weight: Tensor,
    scale: float,
    head_idx: int,
    num_heads: int,
) -> None:
    """Apply scaling factor to a specific head's portion of the weight matrix.

    For combined Q/K projections with shape [num_heads * head_dim, hidden_dim],
    this function scales only the portion corresponding to the specified head.

    Args:
        weight: The weight tensor to scale (modified in-place)
        scale: Scaling factor to apply
        head_idx: Index of the attention head (0-indexed)
        num_heads: Total number of attention heads
    """
    if weight is None:
        return

    # Handle DTensor by getting local tensor
    if hasattr(weight, '_local_tensor'):
        local_weight = weight._local_tensor
    else:
        local_weight = weight

    # Determine the head dimension from weight shape
    # Weight shape is typically [out_features, in_features] where out_features = num_heads * head_dim
    out_features = local_weight.shape[0]
    head_dim = out_features // num_heads

    if head_dim * num_heads != out_features:
        # Weight is not evenly divisible by num_heads, might be a single head or different layout
        # Just scale the entire weight in this case
        local_weight.mul_(scale)
        return

    # Calculate the slice for this head
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    # Scale only the portion of the weight corresponding to this head
    local_weight[start_idx:end_idx].mul_(scale)


# ==================== End QK-Clip Helper Functions ====================


# @torch.compile(fullgraph=True)
def _is_expert_param_name_helper(name: str) -> bool:
    """Helper function to check if parameter name indicates it's an expert parameter."""
    expert_patterns = [
        "experts.",
        ".expert.",
        "expert_",
        "moe.expert",
        "shared_experts",
        "routed_experts",
        ".experts[",
        ".w1.",
        ".w2.",
        ".w3.",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in expert_patterns)


def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
