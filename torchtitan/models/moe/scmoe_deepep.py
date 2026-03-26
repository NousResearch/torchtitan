# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ScMoE with DeepEP backend for efficient EP communication with overlap.

Per the ScMoE paper, only dispatch/combine (all-to-all communication) should
overlap with compute. Expert GEMM stays on the default stream alongside
shared experts, since two compute kernels cannot truly run in parallel
on the same GPU.

The overlap pattern follows standard DeepEPMoE:
1. Router (compute, default stream)
2. Dispatch tokens (sync — DeepEP dispatch waits for completion)
3. Expert GEMM (compute, default stream)
4. Combine tokens (ASYNC — returns immediately, comm runs in background)
5. SharedExperts (compute, default stream — overlaps with combine a2a)
6. sync_combine() — wait for combine to finish
7. Return shared_out + routed_out

The ScMoE twist: routed experts use shortcut input (layer L-1), shared
experts use current input (layer L). Since these are independent inputs,
the shortcut enables the overlap architecture.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.config.job_config import PEFT
from torchtitan.models.moe.moe import (
    FeedForward,
    GroupedExperts,
    moe_init_std,
    MoEArgs,
    TokenChoiceTopKRouter,
    trunc_normal_,
)
from torchtitan.tools.logging import logger


class ScMoEDeepEP(nn.Module):
    """
    ScMoE with DeepEP backend for efficient expert-parallel communication.

    Follows the standard DeepEP overlap pattern: shared experts compute
    overlaps with async combine communication. The ScMoE shortcut connection
    provides independent inputs for the two paths.

    Args:
        moe_args: MoE configuration
        dim: Model dimension
        hidden_dim: MoE expert hidden dimension
        peft_config: Optional PEFT configuration
    """

    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        peft_config: Optional[PEFT] = None,
    ):
        super().__init__()

        num_experts = moe_args.num_experts
        self.num_experts = num_experts
        self.score_before_experts = moe_args.score_before_experts

        # Router for expert selection
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            num_expert_groups=moe_args.num_expert_groups,
            num_limited_groups=moe_args.num_limited_groups,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            gate_bias=moe_args.gate_bias,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )

        # Routed experts (will be wrapped with DeepEPExpertParallel)
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )

        # Shared experts (overlap compute with async combine)
        self.shared_experts = (
            FeedForward(
                dim=dim,
                hidden_dim=hidden_dim * moe_args.num_shared_experts,
                peft_config=peft_config,
            )
            if moe_args.num_shared_experts > 0
            else None
        )

        self.shared_gate = (
            nn.Linear(dim, 1, bias=False) if moe_args.shared_gate else None
        )

        # Layer norms (separate because inputs come from different layers)
        self.routed_norm = nn.RMSNorm(dim)  # For shortcut input → routed experts
        self.shared_norm = nn.RMSNorm(dim)  # For current input → shared experts

        # Load balancing
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

        # Flag to track if DeepEP expert parallel has been applied
        self._deepep_initialized = False

    def forward(
        self,
        x_current: torch.Tensor,
        x_shortcut: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with DeepEP communication overlap.

        All compute runs on the default stream. DeepEP handles dispatch/combine
        communication internally (dispatch is sync, combine is async).
        SharedExperts overlaps with the async combine communication.

        Args:
            x_current: Current layer's post-attention output [bs, slen, dim]
            x_shortcut: Previous layer's output for routing [bs, slen, dim]

        Returns:
            Combined output [bs, slen, dim]
        """
        from torchtitan.distributed.deepep import sync_combine

        bs, slen, dim = x_current.shape
        x_current_flat = x_current.view(-1, dim)
        x_shortcut_flat = x_shortcut.view(-1, dim)

        # Normalize inputs
        x_routed = self.routed_norm(x_shortcut_flat)
        x_shared = self.shared_norm(x_current_flat)

        # Route tokens to experts (on shortcut input)
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_routed, self.expert_bias
        )

        # Update load balancing stats
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Dispatch(sync) → Expert GEMM → Combine(async)
        # DeepEPExpertParallel hooks handle dispatch/combine inside experts()
        # Combine returns immediately — communication runs in background
        routed_output = self.experts(
            x_routed,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.num_experts,
        )

        # SharedExperts on current input — overlaps with async combine a2a
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x_shared)
            if self.shared_gate is not None:
                shared_output = F.sigmoid(self.shared_gate(x_shared)) * shared_output
        else:
            shared_output = None

        # Wait for combine to finish before using routed_output
        sync_combine()

        # Combine outputs
        if shared_output is not None:
            output = routed_output + shared_output
        else:
            output = routed_output

        return output.view(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device, n_layers: int):
        """Initialize weights for all submodules."""
        self.experts.init_weights(init_std, n_layers)
        self.router.init_weights(init_std, n_layers)

        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)
            if self.shared_gate is not None:
                trunc_normal_(
                    self.shared_gate.weight,
                    mean=0.0,
                    std=moe_init_std(self.shared_gate.weight.shape[1], n_layers),
                )

        self.routed_norm.reset_parameters()
        self.shared_norm.reset_parameters()

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.num_experts, dtype=torch.float32
                )

    def pop_expert_routing_metrics(self) -> torch.Tensor | None:
        """Pop and return expert routing metrics for logging."""
        return None


class ScMoEDeepEPTransformerBlock(nn.Module):
    """
    Transformer block with ScMoE + DeepEP for communication hiding.

    Args:
        layer_id: Layer index
        dim: Model dimension
        hidden_dim: MoE expert hidden dimension
        moe_args: MoE configuration
        norm_eps: Epsilon for RMSNorm
        attention_module: Pre-configured attention module
        peft_config: Optional PEFT configuration
        n_layers: Total number of layers
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        hidden_dim: int,
        moe_args: MoEArgs,
        norm_eps: float = 1e-5,
        attention_module: Optional[nn.Module] = None,
        peft_config: Optional[PEFT] = None,
        n_layers: int = 1,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.dim = dim
        self.n_layers = n_layers

        self.attention = attention_module
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)

        self.scmoe = ScMoEDeepEP(
            moe_args=moe_args,
            dim=dim,
            hidden_dim=hidden_dim,
            peft_config=peft_config,
        )

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.shortcut_position = moe_args.scmoe.shortcut_position

    def forward(
        self,
        x: torch.Tensor,
        shortcut_input: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        attention_masks,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_attn = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if shortcut_input is None:
            shortcut_input = x_attn

        scmoe_output = self.scmoe(x_current=x_attn, x_shortcut=shortcut_input)
        output = x_attn + scmoe_output
        next_shortcut = x_attn

        return output, next_shortcut

    def init_weights(self, buffer_device: torch.device):
        """Initialize weights."""
        self.attention_norm.reset_parameters()
        if self.attention is not None and hasattr(self.attention, "init_weights"):
            self.attention.init_weights(self.weight_init_std)
        self.scmoe.init_weights(self.weight_init_std, buffer_device, self.n_layers)


def build_scmoe_deepep(
    args: MoEArgs,
    dim: int,
    hidden_dim: int,
    peft_config: Optional[PEFT] = None,
) -> nn.Module:
    """Factory for ScMoE with DeepEP backend."""
    if not args.use_scmoe:
        raise ValueError("build_scmoe_deepep called but use_scmoe=False")
    if not args.use_deepep:
        raise ValueError("build_scmoe_deepep called but use_deepep=False")

    logger.info(
        f"ScMoE+DeepEP: num_experts={args.num_experts}, top_k={args.top_k}, "
        f"dim={dim}, hidden_dim={hidden_dim}, "
        f"num_shared_experts={args.num_shared_experts}, "
        f"shortcut_position={args.scmoe.shortcut_position}"
    )

    return ScMoEDeepEP(
        moe_args=args,
        dim=dim,
        hidden_dim=hidden_dim,
        peft_config=peft_config,
    )
