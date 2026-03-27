# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shortcut Connected MoE (ScMoE) implementation for communication hiding.

ScMoE rearranges computation order so that:
- Shared experts process layer L's output (current input)
- Routed experts process layer L-1's output (shortcut input)

Since these use independent inputs, the communication required by routed
experts (dispatch/combine all-to-all in EP) can be overlapped with shared
experts compute. Per the paper, ONLY communication goes on the comm stream;
Expert GEMM stays on the default stream alongside shared experts.

Without EP (single-node, no dispatch/combine), ScMoE provides no throughput
benefit — it is purely an architectural change (shortcut connection for
model quality).

With EP (DeepEP), the overlap is: SharedExperts compute || combine a2a.
See scmoe_deepep.py for that implementation.

Reference: "Shortcut-connected Expert Parallelism for Accelerating
Mixture-of-Experts" (arXiv:2404.05019)
"""

import os
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


class ScMoEStreamManager:
    """Manages CUDA streams for ScMoE parallel execution.

    Note: In the paper's design, only dispatch/combine communication goes
    on the comm stream. Expert GEMM and all other compute stay on the
    default stream. For the non-EP case (this module), no stream overlap
    is needed. This class is retained for the DeepEP variant which may
    use it for fine-grained dispatch/combine overlap.
    """

    _instance: Optional["ScMoEStreamManager"] = None

    def __init__(self):
        self.comm_stream: Optional[torch.cuda.Stream] = None
        self._events: dict[str, torch.cuda.Event] = {}

    @classmethod
    def get_instance(cls) -> "ScMoEStreamManager":
        """Get singleton instance of stream manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_comm_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Get or create the communication stream."""
        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream(device=device)
        return self.comm_stream

    def record_event(self, name: str, stream: Optional[torch.cuda.Stream] = None):
        """Record an event on the given stream (default: current stream)."""
        event = torch.cuda.Event()
        if stream is not None:
            event.record(stream)
        else:
            event.record()
        self._events[name] = event

    def wait_event(self, name: str, stream: Optional[torch.cuda.Stream] = None):
        """Make the given stream wait for a recorded event."""
        event = self._events.get(name)
        if event is not None:
            if stream is not None:
                stream.wait_event(event)
            else:
                torch.cuda.current_stream().wait_event(event)

    def sync_comm_to_default(self):
        """Synchronize default stream with comm stream completion."""
        if self.comm_stream is not None:
            event = torch.cuda.Event()
            event.record(self.comm_stream)
            torch.cuda.current_stream().wait_event(event)


class ScMoE(nn.Module):
    """
    Shortcut Connected MoE (non-EP variant).

    Routed experts process the shortcut input (from layer L-1) while shared
    experts process the current input (from layer L). Without EP, all
    computation is sequential on the default stream — the ScMoE benefit is
    purely the architectural change (shortcut connection).

    With EP (see ScMoEDeepEP), shared experts overlap with combine a2a.

    Simulated comm delay (SCMOE_COMM_DELAY_MS env var):
        When set, inserts artificial CUDA sleeps to simulate slow all-to-all.
        With use_separate_streams=True: shared_experts overlaps with dispatch.
        With use_separate_streams=False: everything runs sequentially.
        This demonstrates the overlap benefit without requiring real EP.

    Args:
        moe_args: MoE configuration
        dim: Model dimension
        hidden_dim: MoE expert hidden dimension
        peft_config: Optional PEFT configuration
    """

    # Simulated delays (cycles). ~2M cycles per ms on B200.
    # SCMOE_COMM_DELAY_MS: simulates slow dispatch/combine all-to-all
    # SCMOE_COMPUTE_PAD_MS: pads shared_experts compute to simulate large overlap window
    _comm_delay_cycles: int = int(
        float(os.environ.get("SCMOE_COMM_DELAY_MS", "0")) * 2_000_000
    )
    _compute_pad_cycles: int = int(
        float(os.environ.get("SCMOE_COMPUTE_PAD_MS", "0")) * 2_000_000
    )

    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        peft_config: Optional[PEFT] = None,
    ):
        super().__init__()

        num_experts = moe_args.num_experts
        self.use_separate_streams = moe_args.scmoe.use_separate_streams

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

        # Routed experts (grouped GEMM)
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )

        # Shared experts (overlap compute in EP case)
        self.shared_experts = (
            FeedForward(
                dim=dim,
                hidden_dim=hidden_dim * moe_args.num_shared_experts,
                peft_config=peft_config,
            )
            if moe_args.num_shared_experts > 0
            else None
        )

        # Shared gate for weighted combination
        self.shared_gate = (
            nn.Linear(dim, 1, bias=False) if moe_args.shared_gate else None
        )

        # Layer norms (separate because inputs come from different layers)
        self.routed_norm = nn.RMSNorm(dim)  # For shortcut input → routed experts
        self.shared_norm = nn.RMSNorm(dim)  # For current input → shared experts

        # Token reorderer for expert routing
        from torchtitan.models.moe.moe import TokenReorderer

        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)

        self.score_before_experts = moe_args.score_before_experts
        self.top_k = moe_args.top_k

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

    def _routed_forward(
        self,
        x_routed: torch.Tensor,
        bs: int,
        slen: int,
        dim: int,
    ) -> torch.Tensor:
        """Routed experts forward pass on shortcut input.

        Args:
            x_routed: Normalized shortcut input [bs*slen, dim]
            bs, slen, dim: Original tensor dimensions

        Returns:
            Routed experts output [bs*slen, dim]
        """
        # Route tokens to experts
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_routed, self.expert_bias
        )

        # Update load balancing stats
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Reorder tokens for grouped GEMM
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # Gather routed tokens
        routed_input = x_routed[token_indices_experts_sorted // self.top_k]

        # Apply routing scores before experts if configured
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x_routed.dtype)

        # Expert computation (GEMM — always on default stream)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(-1, self.top_k, dim)

        # Combine routed outputs
        if not self.score_before_experts:
            out_routed = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x_routed.dtype)
                .squeeze(1)
            )
        else:
            out_routed = routed_output_unsorted.sum(dim=1)

        return out_routed

    def _shared_forward(self, x_shared: torch.Tensor) -> Optional[torch.Tensor]:
        """Shared experts forward pass on current input.

        Args:
            x_shared: Normalized current layer input [bs*slen, dim]

        Returns:
            Shared experts output [bs*slen, dim], or None if no shared experts.
        """
        if self.shared_experts is None:
            return None

        shared_output = self.shared_experts(x_shared)

        # Apply shared gate if configured
        if self.shared_gate is not None:
            shared_output = F.sigmoid(self.shared_gate(x_shared)) * shared_output

        return shared_output

    # Class-level timing control (enable with SCMOE_TIMING=1)
    _timing_enabled: bool = os.environ.get("SCMOE_TIMING", "0") == "1"
    _timing_step: int = 0
    _timing_log_interval: int = 5  # Log every N forward calls

    @classmethod
    def enable_timing(cls, enabled: bool = True, log_interval: int = 10):
        cls._timing_enabled = enabled
        cls._timing_log_interval = log_interval

    def forward(
        self,
        x_current: torch.Tensor,
        x_shortcut: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with shortcut connection.

        When SCMOE_COMM_DELAY_MS is set and use_separate_streams=True, simulates
        the paper's overlap strategy:
            Comm stream:    [Dispatch(sleep)]  ..........  [Combine(sleep)]
            Default stream: [SharedExperts] → sync → [Expert GEMM] → sync combine

        When use_separate_streams=False (or no delay), runs sequentially.

        Args:
            x_current: Current layer's post-attention output [bs, slen, dim]
            x_shortcut: Previous layer's output for routing [bs, slen, dim]

        Returns:
            Combined output [bs, slen, dim] = shared_out + routed_out
        """
        bs, slen, dim = x_current.shape
        x_current_flat = x_current.view(-1, dim)
        x_shortcut_flat = x_shortcut.view(-1, dim)

        x_routed = self.routed_norm(x_shortcut_flat)
        x_shared = self.shared_norm(x_current_flat)

        comm_delay = self._comm_delay_cycles
        compute_pad = self._compute_pad_cycles

        if comm_delay > 0 and self.use_separate_streams and x_current.is_cuda:
            # --- ScMoE overlap: shared_experts || dispatch, then GEMM, then combine ---
            #
            # Comm stream:    [Dispatch(sleep)] ............. [Combine(sleep)]
            # Default stream: [SharedExperts+pad] → sync → [Expert GEMM] → sync
            stream_mgr = ScMoEStreamManager.get_instance()
            comm_stream = stream_mgr.get_comm_stream(x_current.device)

            # Async dispatch (simulated) on comm_stream
            ready = torch.cuda.Event()
            ready.record()
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                torch.cuda._sleep(comm_delay)
                dispatch_done = torch.cuda.Event()
                dispatch_done.record(comm_stream)

            # SharedExperts on default stream — overlaps with dispatch
            shared_out = self._shared_forward(x_shared)
            if compute_pad > 0:
                torch.cuda._sleep(compute_pad)  # pad to simulate larger compute

            # Wait for dispatch, then Expert GEMM on default stream
            torch.cuda.current_stream().wait_event(dispatch_done)
            routed_out = self._routed_forward(x_routed, bs, slen, dim)

            # Async combine (simulated) on comm_stream
            gemm_done = torch.cuda.Event()
            gemm_done.record()
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(gemm_done)
                torch.cuda._sleep(comm_delay)
                combine_done = torch.cuda.Event()
                combine_done.record(comm_stream)

            # Wait for combine
            torch.cuda.current_stream().wait_event(combine_done)

        elif comm_delay > 0:
            # --- Sequential: dispatch, GEMM, combine, shared (no overlap) ---
            torch.cuda._sleep(comm_delay)  # dispatch
            routed_out = self._routed_forward(x_routed, bs, slen, dim)
            torch.cuda._sleep(comm_delay)  # combine
            shared_out = self._shared_forward(x_shared)
            if compute_pad > 0:
                torch.cuda._sleep(compute_pad)  # same compute pad, but sequential

        else:
            # --- No delay: pure sequential ---
            routed_out = self._routed_forward(x_routed, bs, slen, dim)
            shared_out = self._shared_forward(x_shared)

        if shared_out is not None:
            output = shared_out + routed_out
        else:
            output = routed_out

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

        # Reset norms
        self.routed_norm.reset_parameters()
        self.shared_norm.reset_parameters()

        # Initialize buffers
        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )

    def pop_expert_routing_metrics(self) -> torch.Tensor | None:
        """Pop and return expert routing metrics for logging."""
        return None


class ScMoETransformerBlock(nn.Module):
    """
    Transformer block with ScMoE for communication hiding.

    This block implements the shortcut connection architecture where:
    - Attention processes the current input
    - Routed experts process the shortcut input (from layer L-1)
    - Shared experts process the current post-attention output
    - With EP, shared experts overlap with combine communication

    Args:
        layer_id: Layer index (used for weight initialization)
        dim: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per attention head
        hidden_dim: MoE expert hidden dimension
        moe_args: MoE configuration
        norm_eps: Epsilon for RMSNorm
        attention_module: Pre-configured attention module
        peft_config: Optional PEFT configuration
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
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

        self.scmoe = ScMoE(
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
        """
        Forward pass with shortcut connection for ScMoE.

        Args:
            x: Input tensor [bs, slen, dim]
            shortcut_input: Previous layer's output for routed experts [bs, slen, dim]
                           If None (first ScMoE layer), uses x as shortcut
            freqs_cis: RoPE frequencies
            attention_masks: Attention masks
            positions: Optional position indices for RoPE

        Returns:
            Tuple of:
            - output: Layer output [bs, slen, dim]
            - next_shortcut: Output to use as shortcut for next layer [bs, slen, dim]
        """
        x_attn = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if shortcut_input is None:
            shortcut_input = x_attn

        scmoe_output = self.scmoe(x_current=x_attn, x_shortcut=shortcut_input)
        output = x_attn + scmoe_output

        if self.shortcut_position == "pos1":
            next_shortcut = x_attn
        else:
            next_shortcut = x_attn

        return output, next_shortcut

    def init_weights(self, buffer_device: torch.device):
        """Initialize weights for all submodules."""
        self.attention_norm.reset_parameters()
        if self.attention is not None and hasattr(self.attention, "init_weights"):
            self.attention.init_weights(self.weight_init_std)
        self.scmoe.init_weights(self.weight_init_std, buffer_device, self.n_layers)


def build_scmoe(
    args: MoEArgs,
    dim: int,
    hidden_dim: int,
    peft_config: Optional[PEFT] = None,
) -> nn.Module:
    """Factory for ScMoE modules.

    Args:
        args: MoE configuration (must have use_scmoe=True)
        dim: Model dimension
        hidden_dim: Expert hidden dimension
        peft_config: Optional PEFT configuration

    Returns:
        ScMoE module
    """
    if not args.use_scmoe:
        raise ValueError("build_scmoe called but use_scmoe=False")

    logger.info(
        f"ScMoE: num_experts={args.num_experts}, top_k={args.top_k}, "
        f"dim={dim}, hidden_dim={hidden_dim}, "
        f"num_shared_experts={args.num_shared_experts}, "
        f"shortcut_position={args.scmoe.shortcut_position}"
    )

    return ScMoE(
        moe_args=args,
        dim=dim,
        hidden_dim=hidden_dim,
        peft_config=peft_config,
    )
