# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

import torch
from torch import nn

from torchtitan.distributed.deepep import sync_combine

from .moe import GroupedExperts, MoE, MoEArgs


class DeepEPMoE(MoE):
    """
    Mixture of Experts with DeepEP communication.

    Inherits from MoE but overrides forward() to pass routing info to experts,
    letting DeepEPExpertParallel hooks handle dispatch/combine.

    The forward pass is structured to overlap shared_experts computation with
    the DeepEP combine communication:
    1. Router computes expert assignments
    2. DeepEP dispatches tokens to experts (sync)
    3. Experts process tokens
    4. DeepEP combine starts (async) - returns immediately
    5. shared_experts runs IN PARALLEL with combine communication
    6. sync_combine() waits for combine to complete
    7. Addition of shared_experts output and routed_output
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DeepEP communication and overlapped shared_experts.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions. The combine operation runs
        asynchronously, allowing shared_experts to overlap with the
        combine all-to-all communication.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Call experts with routing info - hooks handle DeepEP dispatch/combine.
        # The combine operation returns asynchronously, allowing overlap with
        # shared_experts computation below.
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        # shared_experts runs in parallel with combine communication.
        # This is the key optimization - we overlap compute with communication.
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Sync the combine operation before using routed_output.
        # This inserts a CUDA stream wait, ensuring combine is complete before
        # the subsequent addition or reshape operations read routed_output.
        sync_combine()

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)


class CommOnlyExperts(GroupedExperts):
    """Experts module that skips GEMM — only exists so EP hooks (dispatch/combine) fire.

    Has the same w1/w2/w3 parameters as GroupedExperts so that:
    - DeepEPExpertParallel._partition_fn can shard them on dim 0
    - DeepEPExpertParallel._token_dispatch reads mod.w1 for num_local_experts
    - FSDP can manage them normally
    But forward() returns zeros instead of computing SwiGLU.
    """

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Return zeros with same shape — autograd-safe via multiply
        return (x[..., :1] * 0).expand_as(x)


class CommOnlyMoE(DeepEPMoE):
    """Profiling MoE: router + DeepEP dispatch/combine only.

    No expert GEMM, no shared expert. Isolates the cost of:
    - Token routing (gate linear + topk)
    - DeepEP all-to-all dispatch and combine
    - Permute/unpermute tokens
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        # Replace GroupedExperts with comm-only version (no GEMM)
        self.experts = CommOnlyExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        # Remove shared expert entirely
        self.shared_experts = None
        self.shared_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # Route tokens to experts
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # EP hooks fire: dispatch → CommOnlyExperts (returns zeros) → combine
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        # Sync the async combine
        sync_combine()

        return routed_output.reshape(bs, slen, dim)


class EmptyExperts(GroupedExperts):
    """Experts module that does nothing. Has w1/w2/w3 params for FSDP/EP hooks."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (x[..., :1] * 0).expand_as(x)


class EmptyMoE(DeepEPMoE):
    """Profiling MoE: does absolutely nothing. Returns zeros.

    Still has the full module structure (router, experts with params) so that
    FSDP and EP parallelization hooks are applied normally. But forward()
    skips all computation — no routing, no dispatch, no GEMM, no combine.
    This isolates the pure framework overhead (FSDP all-gather/reduce-scatter
    on MoE params, layer iteration, etc.) from actual MoE work.
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim, hidden_dim)
        self.experts = EmptyExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.shared_experts = None
        self.shared_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do nothing — just return zeros with correct shape
        return x * 0
