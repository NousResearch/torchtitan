# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DeviceMesh

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims

# Import the Muon optimizer (assuming it's available)
from .muon import Muon, QKClipConfig, AttentionHeadParams
from .parameter_classification import (
    create_parameter_groups_with_attention,
    collect_attention_layers,
    AttentionLayerInfo,
)

__all__ = [
    "MuonOptimizersContainer",
    "build_muon_optimizers",
    "MuonOptimizerConfig",
    "QKClipConfig",
    "AttentionHeadParams",
    "AttentionLayerInfo",
]


@dataclass
class MuonOptimizerConfig:
    """Extended optimizer config for Muon-specific parameters."""

    # Standard optimizer parameters
    name: str = "muon"
    lr: float = 0.01
    weight_decay: float = 0.01

    # Muon-specific parameters
    mu: float = 0.95  # Momentum for Muon
    betas: tuple[float, float] = (0.9, 0.95)  # Betas for AdamW and Lion
    epsilon: float = 1e-8
    nesterov: bool = False  # Whether to use Nesterov momentum
    adjust_lr: Optional[str] = "spectral_norm"  # "spectral_norm", "rms_norm", or None
    flatten: bool = False  # Whether to flatten 3D+ tensors to 2D
    use_triton: bool = False  # Whether to use Triton kernel for Newton-Schulz

    # Algorithm selection per parameter group
    # Can be "muon", "adamw", or "lion"
    algorithm: str = "muon"

    # Parameter-specific optimizer selection
    scalar_optimizer: str = "adamw"  # For 1D parameters (biases, layer norms)
    embedding_optimizer: str = "adamw"  # For embedding layers
    head_optimizer: str = "adamw"  # For model head/output layers
    routing_optimizer: Optional[str] = None  # For routing layers (DeepSeek MoE)
    expert_optimizer: Optional[str] = None  # For expert weights (MoE experts)

    # Additional optimizer options
    head_lr_scaling: bool = True  # Apply 1/sqrt(dim) scaling to head layers

    # Learning rate scaling factors
    scalar_lr_factor: float = 1.0  # LR multiplier for scalar parameters
    embedding_lr_factor: float = 1.0  # LR multiplier for embedding parameters
    head_lr_factor: float = (
        1.0  # LR multiplier for head parameters (after head_lr_scaling)
    )
    routing_lr_factor: float = 1.0  # LR multiplier for routing parameters
    expert_lr_factor: float = 1.0  # LR multiplier for expert parameters

    # Gradient synchronization
    replicate_mesh_grad_sync: bool = True

    # QK-Clip parameters (MuonClip from Kimi K2)
    # See: https://arxiv.org/abs/2507.20534
    qk_clip_enabled: bool = False  # Whether to enable QK-Clip for training stability
    qk_clip_tau: float = 100.0  # Threshold τ for attention logit clipping
    qk_clip_interval: int = 1  # How often to apply QK-Clip (1 = every step, 10 = every 10 steps)
    # Note: MLA mode is auto-detected based on whether MLA params (wq_c, wk_c, etc.) are found


class MuonOptimizersContainer(OptimizersContainer):
    """A container for Muon optimizers compatible with TorchTitan interface.

    This class wraps the Muon optimizer to make it compatible with the
    TorchTitan OptimizersContainer interface while preserving Muon's
    distributed training capabilities.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        muon_config (MuonOptimizerConfig): Configuration for Muon optimizer.
        parallel_dims (ParallelDims): Parallel dimensions configuration.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        muon_config: MuonOptimizerConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        self.model_parts = model_parts
        self.muon_config = muon_config
        self.parallel_dims = parallel_dims

        # Setup device meshes from parallel dimensions
        distributed_mesh = self._setup_device_mesh(parallel_dims)

        # Classify parameters and collect attention layer info for QK-Clip
        classification_result = create_parameter_groups_with_attention(model_parts, muon_config)
        param_groups = classification_result.param_groups
        self._attention_layers = classification_result.attention_layers

        # Create QK-Clip configuration if enabled
        qk_clip_config = None
        if muon_config.qk_clip_enabled:
            qk_clip_config = QKClipConfig(
                enabled=muon_config.qk_clip_enabled,
                tau=muon_config.qk_clip_tau,
            )

        # Create the Muon optimizer
        self.muon_optimizer = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=muon_config.lr,
            mu=muon_config.mu,
            betas=muon_config.betas,
            weight_decay=muon_config.weight_decay,
            epsilon=muon_config.epsilon,
            nesterov=muon_config.nesterov,
            adjust_lr=muon_config.adjust_lr,
            flatten=muon_config.flatten,
            use_triton=muon_config.use_triton,
            qk_clip_config=qk_clip_config,
        )

        # Initialize parent class with dummy optimizer kwargs
        # This ensures hooks and other functionality work
        super().__init__(
            model_parts=model_parts,
            optimizer_cls=torch.optim.SGD,  # Dummy, not used
            optimizer_kwargs={"lr": muon_config.lr},  # Dummy, not used
        )

        # For compatibility with OptimizersContainer interface
        self.optimizers = [self.muon_optimizer]

        # Auto-register attention layers for QK-Clip using classification results
        if muon_config.qk_clip_enabled and self._attention_layers:
            self._register_attention_layers_from_classification()

        # Track max attention logit for logging (updated after each step)
        self._last_max_logit: Optional[float] = None
        # Step counter for QK-Clip interval tracking
        self._qk_clip_step: int = 0

    def _setup_device_mesh(
        self, parallel_dims: ParallelDims
    ) -> Optional[Union[DeviceMesh, ProcessGroup]]:
        """Setup device mesh based on parallel dimensions.

        For Muon, we use the dp_shard mesh for distributed communication.
        """
        distributed_mesh = None

        # Get the world mesh from parallel_dims
        world_mesh = parallel_dims.world_mesh

        # For Muon, we primarily use the dp_shard mesh for distributed operations
        if parallel_dims.dp_shard_enabled:
            # Extract the dp_shard submesh
            if "dp_shard" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_shard"]
            elif "dp_shard_cp" in world_mesh.mesh_dim_names:
                # If context parallel is enabled, use dp_shard_cp mesh
                distributed_mesh = world_mesh["dp_shard_cp"]
        elif parallel_dims.dp_replicate_enabled:
            # If no dp_shard but dp_replicate is enabled, use that
            if "dp_replicate" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_replicate"]
            elif "dp" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp"]

        return distributed_mesh

    def __iter__(self):
        """Iterate over optimizers for compatibility."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers."""
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        """Perform optimization step."""
        self.muon_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients for all optimizers."""
        # Call parent class method to ensure all optimizers in self.optimizers are handled
        super().zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict using distributed checkpoint utilities."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(
                func, self.model_parts, [self.muon_optimizer] * len(self.model_parts)
            )
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict using distributed checkpoint utilities."""
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, [self.muon_optimizer] * len(self.model_parts)))

    # ==================== QK-Clip (MuonClip) Methods ====================

    def post_build_setup(self, model_parts: List[nn.Module]) -> None:
        """Setup QK-Clip after optimizer is built.

        This method enables max logit tracking on model attention modules and
        registers a post-step hook to apply QK-Clip based on qk_clip_interval.

        Args:
            model_parts: List of model parts to setup.
        """
        if not self.muon_config.qk_clip_enabled:
            return

        interval = self.muon_config.qk_clip_interval

        # Enable max logit tracking for the first step (step 0 is a clip step)
        self._enable_max_logit_tracking(model_parts, enabled=True)

        # Register post-step hook for QK-Clip
        def qk_clip_post_step_hook(*args, **kwargs):
            # Check if this is a clip step (before incrementing counter)
            is_clip_step = (self._qk_clip_step % interval) == 0

            if is_clip_step:
                # Collect max logits from model attention modules
                attention_max_logits = self._collect_attention_max_logits(model_parts)

                if attention_max_logits:
                    # Compute and store global max logit for logging
                    global_max = None
                    for layer_logits in attention_max_logits.values():
                        # layer_logits shape: (batch_size, n_heads) - take max across batch and heads
                        layer_max = layer_logits.max().item()
                        if global_max is None or layer_max > global_max:
                            global_max = layer_max
                    self._last_max_logit = global_max

                    # Apply QK-Clip
                    self.apply_qk_clip(attention_max_logits)

                # Clear max logits
                self._clear_attention_max_logits(model_parts)

            # Increment step counter
            self._qk_clip_step += 1

            # Enable/disable tracking for next step based on interval
            next_is_clip_step = (self._qk_clip_step % interval) == 0
            self._enable_max_logit_tracking(model_parts, enabled=next_is_clip_step)

        self.register_step_post_hook(qk_clip_post_step_hook)

    def _enable_max_logit_tracking(
        self, model_parts: List[nn.Module], enabled: bool = True
    ) -> None:
        """Enable or disable max logit tracking on model attention modules.

        Args:
            model_parts: List of model parts.
            enabled: Whether to enable (True) or disable (False) tracking.
        """
        for model in model_parts:
            if hasattr(model, "set_track_max_logits"):
                model.set_track_max_logits(enabled)

    def _collect_attention_max_logits(
        self, model_parts: List[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Collect max attention logits from model attention modules.

        Returns a dictionary mapping layer names to max logit tensors.
        """
        attention_max_logits = {}
        for model in model_parts:
            if hasattr(model, "get_attention_max_logits"):
                logits = model.get_attention_max_logits()
                attention_max_logits.update(logits)
        return attention_max_logits

    def _clear_attention_max_logits(self, model_parts: List[nn.Module]) -> None:
        """Clear max logit tracking on model attention modules."""
        for model in model_parts:
            if hasattr(model, "clear_attention_max_logits"):
                model.clear_attention_max_logits()

    def register_attention_params(
        self,
        layer_name: str,
        wq: Optional[torch.Tensor] = None,
        wk: Optional[torch.Tensor] = None,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        wq_c: Optional[torch.Tensor] = None,
        wk_c: Optional[torch.Tensor] = None,
        wq_r: Optional[torch.Tensor] = None,
        wk_r: Optional[torch.Tensor] = None,
    ) -> None:
        """Register attention parameters for QK-Clip.

        Delegates to the underlying Muon optimizer's register_attention_params method.

        Args:
            layer_name: Unique identifier for the attention layer
            wq: Query projection weights
            wk: Key projection weights
            num_heads: Number of attention heads
            head_dim: Dimension per head (inferred from wq if not provided)
            wq_c: (MLA) Query compression weights
            wk_c: (MLA) Key compression weights
            wq_r: (MLA) Query rotary weights (head-specific)
            wk_r: (MLA) Key rotary weights (shared, not clipped)
        """
        self.muon_optimizer.register_attention_params(
            layer_name=layer_name,
            wq=wq,
            wk=wk,
            num_heads=num_heads,
            head_dim=head_dim,
            wq_c=wq_c,
            wk_c=wk_c,
            wq_r=wq_r,
            wk_r=wk_r,
        )

    def apply_qk_clip(
        self,
        attention_max_logits: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply QK-Clip after optimizer step.

        This method should be called AFTER the optimizer step, providing the maximum
        attention logits observed during the forward pass.

        Args:
            attention_max_logits: Dictionary mapping layer names to tensors of max logits
                                  per head. Shape: [num_heads] or scalar.

        Returns:
            Dictionary mapping layer names to the scaling factors applied (γ_h).
        """
        return self.muon_optimizer.apply_qk_clip(attention_max_logits)

    @property
    def qk_clip_enabled(self) -> bool:
        """Check if QK-Clip is enabled."""
        return self.muon_optimizer.qk_clip_enabled

    @property
    def qk_clip_tau(self) -> float:
        """Get the QK-Clip threshold τ."""
        return self.muon_optimizer.qk_clip_tau

    @property
    def last_max_logit(self) -> Optional[float]:
        """Get the max attention logit from the last forward pass.

        This returns the global maximum across all layers, heads, and batch samples.
        Useful for logging/monitoring attention logit growth during training.
        Returns None if no logits have been tracked yet.
        """
        return self._last_max_logit

    def _register_attention_layers_from_classification(self) -> None:
        """Register attention layers detected during parameter classification.

        This is called automatically during __init__ when QK-Clip is enabled.
        Uses the AttentionLayerInfo collected by create_parameter_groups_with_attention.
        """
        for layer_info in self._attention_layers:
            self.muon_optimizer.register_attention_params(
                layer_name=layer_info.layer_name,
                wq=layer_info.wq,
                wk=layer_info.wk,
                num_heads=layer_info.num_heads,
                head_dim=layer_info.head_dim,
                wq_c=layer_info.wq_c,
                wk_c=layer_info.wk_c,
                wq_r=layer_info.wq_r,
                wk_r=layer_info.wk_r,
            )

    def auto_register_attention_params(self) -> None:
        """Manually trigger attention parameter registration.

        Note: This is typically not needed as registration happens automatically
        during optimizer initialization when qk_clip_enabled=True. This method
        is provided for cases where you want to re-scan or update registrations.
        """
        if not self.muon_config.qk_clip_enabled:
            return

        # Use the classification system to detect attention layers
        # MLA params are auto-detected if present
        self._attention_layers = collect_attention_layers(self.model_parts)
        self._register_attention_layers_from_classification()

    @property
    def attention_layers(self) -> List[AttentionLayerInfo]:
        """Get the list of detected attention layers."""
        return self._attention_layers

    # ==================== End QK-Clip Methods ====================


def build_muon_optimizers(
    model_parts: List[nn.Module],
    muon_config: MuonOptimizerConfig,
    parallel_dims: ParallelDims,
) -> MuonOptimizersContainer:
    """Create a MuonOptimizersContainer for the given model parts and config.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        muon_config (MuonOptimizerConfig): Muon optimizer configuration.
        parallel_dims (ParallelDims): Parallel dimensions for the model.

    Returns:
        MuonOptimizersContainer: Container with Muon optimizer.
    """
    return MuonOptimizersContainer(
        model_parts=model_parts,
        muon_config=muon_config,
        parallel_dims=parallel_dims,
    )


def build_optimizers_with_muon_support(
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    muon_config: Optional[MuonOptimizerConfig] = None,
) -> OptimizersContainer:
    """Extended build_optimizers function with Muon support.

    This is a drop-in replacement for the original build_optimizers function
    that adds support for the Muon optimizer.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Standard optimizer config.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
        muon_config (Optional[MuonOptimizerConfig]): Muon-specific config.
            If provided, will use Muon optimizer instead of standard optimizers.

    Returns:
        OptimizersContainer: Container with appropriate optimizer(s).
    """
    # If Muon config is provided, use Muon optimizer
    if muon_config is not None:
        return build_muon_optimizers(model_parts, muon_config, parallel_dims)

    # Otherwise, fall back to original build_optimizers logic
    from torchtitan.components.optimizer import build_optimizers

    return build_optimizers(model_parts, optimizer_config, parallel_dims)


# Example usage and parameter group configuration utilities
class MuonParameterGroupManager:
    """Utility class to manage different algorithms for different parameter groups."""

    @staticmethod
    def create_mixed_param_groups(
        model_parts: List[nn.Module],
        muon_config: MuonOptimizerConfig,
        layer_algorithm_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with different algorithms for different layers.

        Args:
            model_parts: List of model parts
            muon_config: Base configuration
            layer_algorithm_map: Mapping from layer name patterns to algorithms
                                Example: {"attention": "muon", "mlp": "adamw", "embed": "lion"}

        Returns:
            List of parameter group dictionaries
        """
        if layer_algorithm_map is None:
            layer_algorithm_map = {"": "muon"}  # Default to muon for all

        param_groups = []

        for model in model_parts:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine algorithm based on layer name
                algorithm = "muon"  # default
                for pattern, algo in layer_algorithm_map.items():
                    if pattern in name:
                        algorithm = algo
                        break

                # Create parameter group
                param_group = {
                    "params": [param],
                    "algorithm": algorithm,
                    "lr": muon_config.lr,
                    "mu": muon_config.mu,
                    "beta1": muon_config.betas[0],
                    "beta2": muon_config.betas[1],
                    "weight_decay": muon_config.weight_decay,
                    "epsilon": muon_config.epsilon,
                    "nesterov": muon_config.nesterov,
                    "adjust_lr": muon_config.adjust_lr,
                    "flatten": muon_config.flatten,
                }
                param_groups.append(param_group)

        return param_groups


# Example configuration for different model architectures
def get_llama_muon_config() -> MuonOptimizerConfig:
    """Example Muon configuration optimized for LLaMA-style models."""
    return MuonOptimizerConfig(
        name="muon",
        lr=3e-4,
        weight_decay=0.1,
        mu=0.95,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        nesterov=False,
        adjust_lr="spectral_norm",  # For learning rate transfer across model scale
        flatten=False,  # Keep False for transformer attention layers
        use_triton=False,  # Conservative default
        algorithm="muon",
    )


def get_mixed_algorithm_config() -> tuple[MuonOptimizerConfig, Dict[str, str]]:
    """Example configuration using different algorithms for different layers."""
    config = MuonOptimizerConfig(
        name="mixed",
        lr=3e-4,
        weight_decay=0.1,
    )

    # Use Muon for attention layers, AdamW for embeddings, Lion for MLP
    algorithm_map = {
        "attention": "muon",
        "embed": "adamw",
        "mlp": "lion",
    }

    return config, algorithm_map
