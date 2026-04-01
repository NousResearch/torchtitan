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


class ScMoEDeepEP(nn.Module):
    """
    ScMoE with DeepEP backend — paper-exact overlap strategy.

    The overlap follows the paper's Figure 6/7:

        Default stream: [Attention] → wait → [Expert GEMM] → [SharedExperts] → wait
        Comm stream:    [Dispatch a2a......]                  [Combine a2a.....]

    - Dispatch (communication) overlaps with Attention (computation)
    - Combine (communication) overlaps with SharedExperts (computation)
    - Expert GEMM runs on default stream after dispatch completes

    The TransformerBlock calls phased methods to interleave dispatch with
    attention.  For the first ScMoE layer (no shortcut yet), falls back to
    the sequential path via forward().

    Note: self.experts is wrapped by DeepEPExpertParallel (for weight sharding).
    We call self.experts.forward() to bypass the dispatch/combine hooks and
    handle them manually for fine-grained overlap control.

    Args:
        moe_args: MoE configuration
        dim: Model dimension
        hidden_dim: MoE expert hidden dimension
        peft_config: Optional PEFT configuration
    """

    # Enable with SCMOE_TIMING=1. Logs per-layer timing every N forward calls.
    _timing_enabled: bool = os.environ.get("SCMOE_TIMING", "0") == "1"
    _timing_step: int = 0
    _timing_log_interval: int = 10  # Log every N calls (1 step = 60 layers)

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

    # ------------------------------------------------------------------ #
    #  Phased API — called by TransformerBlock for paper-exact overlap    #
    # ------------------------------------------------------------------ #

    _cached_ep_info: tuple | None = None

    def _get_ep_info(self):
        """Get EP group and local expert count from the parallelized experts."""
        if self._cached_ep_info is not None:
            return self._cached_ep_info

        if isinstance(self.experts.w1, DTensor):
            num_local_experts = self.experts.w1.to_local().shape[0]
            mesh = self.experts.w1.device_mesh
            # Find the EP dimension: the one whose group has size > 1
            # (experts are sharded on the EP dim)
            ep_group = None
            for dim in range(mesh.ndim):
                g = mesh.get_group(mesh_dim=dim)
                if g.size() > 1:
                    ep_group = g
                    break
            if ep_group is None:
                # Fallback: 1D mesh
                ep_group = mesh.get_group() if mesh.ndim == 1 else None
        else:
            num_local_experts = self.experts.w1.shape[0]
            ep_group = None

        self._cached_ep_info = (num_local_experts, ep_group)
        return num_local_experts, ep_group

    _sms_configured: bool = False

    def prepare_dispatch(self, x_shortcut: torch.Tensor):
        """Phase 1a: Route + layout on DEFAULT stream (Encode in paper).

        This is all compute — runs on the default stream before overlap starts.
        CPU submits these kernels and returns quickly (non-blocking).
        """
        from torchtitan.distributed.deepep.deepep import get_buffer, get_hidden_bytes

        # Configure DeepEP num_sms once (from env var)
        if not ScMoEDeepEP._sms_configured:
            num_sms_str = os.environ.get("DEEPEP_NUM_SMS", "")
            if num_sms_str:
                from deep_ep import Buffer
                num_sms = int(num_sms_str)
                Buffer.set_num_sms(num_sms)
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                if rank == 0:
                    logger.info(f"[ScMoE] DeepEP num_sms set to {num_sms} (from DEEPEP_NUM_SMS)")
            ScMoEDeepEP._sms_configured = True

        bs, slen, dim = x_shortcut.shape
        x_shortcut_flat = x_shortcut.view(-1, dim)
        x_routed = self.routed_norm(x_shortcut_flat)

        # Router on default stream
        top_scores, selected_experts_indices, _ = self.router(
            x_routed, self.expert_bias
        )

        # Mask + type conversion on default stream (Encode)
        selected_experts_indices = selected_experts_indices.contiguous()
        top_scores = top_scores.contiguous()
        selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
        if top_scores.dtype != torch.float32:
            top_scores = top_scores.float()

        # Layout computation on default stream (Encode)
        num_local_experts, ep_group = self._get_ep_info()
        buffer = get_buffer(ep_group, get_hidden_bytes(x_routed))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert_dispatch,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(
            topk_idx=selected_experts_indices,
            num_experts=self.num_experts,
        )

        # Save everything for the a2a phase
        self._prep = {
            "x_routed": x_routed,
            "selected_experts_indices": selected_experts_indices,
            "top_scores": top_scores,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert_dispatch": num_tokens_per_expert_dispatch,
        }
        self._shape = (bs, slen, dim)

        # Update load balancing
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                num_tokens_per_expert = torch.histc(
                    selected_experts_indices.float().view(-1),
                    bins=self.num_experts,
                    min=0,
                    max=self.num_experts - 1,
                )
                self.tokens_per_expert.add_(num_tokens_per_expert)

    def launch_dispatch_a2a(self, comm_stream: torch.cuda.Stream):
        """Phase 1b: Launch ONLY the a2a transfer on comm_stream.

        This is the ONLY part that goes on comm_stream. The CPU may block here
        (DeepEP spin-loop for recv_count), but attention kernels are already
        queued on the default stream from the caller.
        """
        from torchtitan.distributed.deepep.deepep import _permute_tokens

        prep = self._prep

        # Record that default stream prep (layout + attention) is done
        prep_done = torch.cuda.Event()
        prep_done.record()

        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(prep_done)

            # Pure a2a dispatch (may block CPU in spin-loop)
            (
                recv_x,
                dispatched_indices,
                dispatched_expert_scores,
                num_tokens_per_expert,
                handle_id,
            ) = torch.ops.deepep.dispatch(
                prep["x_routed"],
                prep["selected_experts_indices"],
                prep["top_scores"],
                prep["num_tokens_per_rank"],
                prep["num_tokens_per_rdma_rank"],
                prep["is_token_in_rank"],
                prep["num_tokens_per_expert_dispatch"],
            )

            dispatch_done = torch.cuda.Event()
            dispatch_done.record(comm_stream)

        # Save dispatch results (these tensors are on comm_stream)
        self._dispatch_results = {
            "recv_x": recv_x,
            "dispatched_indices": dispatched_indices,
            "dispatched_expert_scores": dispatched_expert_scores,
            "num_tokens_per_expert": num_tokens_per_expert,
            "handle_id": handle_id,
        }
        self._dispatch_done_event = dispatch_done
        del self._prep

    def wait_dispatch_permute_and_gemm(self) -> torch.Tensor:
        """Phase 2: Wait for a2a, Encode (permute), Expert GEMM — all on default stream."""
        from torchtitan.distributed.deepep.deepep import (
            _permute_tokens,
            DispatchState,
        )

        # Wait for a2a to complete
        torch.cuda.current_stream().wait_event(self._dispatch_done_event)

        r = self._dispatch_results
        num_recv_tokens = r["recv_x"].shape[0]

        # Permute (Encode) on DEFAULT stream — this is compute, not comm
        hidden_states, permuted_scores, permuted_indices = _permute_tokens(
            r["recv_x"], r["dispatched_indices"], r["dispatched_expert_scores"]
        )

        num_tokens_per_expert = r["num_tokens_per_expert"].to(hidden_states.device)

        if self.score_before_experts and permuted_scores is not None:
            hidden_states = hidden_states * permuted_scores.to(
                hidden_states.dtype
            ).reshape(-1, 1)
            permuted_scores_for_state = None
        else:
            permuted_scores_for_state = permuted_scores

        self._dispatch_state = DispatchState(
            handle_id=r["handle_id"],
            permuted_indices=permuted_indices,
            num_recv_tokens=num_recv_tokens,
            permuted_scores=permuted_scores_for_state,
        )
        del self._dispatch_results, self._dispatch_done_event

        # Expert GEMM on default stream — bypass hooks
        routed_output = self.experts.forward(hidden_states, num_tokens_per_expert)
        return routed_output

    # Keep old start_dispatch for backward compat (used by tests)
    def start_dispatch(self, x_shortcut: torch.Tensor, comm_stream: torch.cuda.Stream):
        """Route + dispatch on comm_stream (original working version).

        All of dispatch_tokens runs on comm_stream. No intermediate _prep dict.
        """
        from torchtitan.distributed.deepep import dispatch_tokens

        # Configure num_sms once
        if not ScMoEDeepEP._sms_configured:
            num_sms_str = os.environ.get("DEEPEP_NUM_SMS", "")
            if num_sms_str:
                from deep_ep import Buffer
                Buffer.set_num_sms(int(num_sms_str))
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                if rank == 0:
                    logger.info(f"[ScMoE] DeepEP num_sms set to {num_sms_str}")
            ScMoEDeepEP._sms_configured = True

        bs, slen, dim = x_shortcut.shape
        x_shortcut_flat = x_shortcut.view(-1, dim)
        x_routed = self.routed_norm(x_shortcut_flat)

        # Router on default stream
        top_scores, selected_experts_indices, _ = self.router(
            x_routed, self.expert_bias
        )

        num_local_experts, ep_group = self._get_ep_info()

        # Record routing done
        routing_done = torch.cuda.Event()
        routing_done.record()

        # ALL of dispatch (layout + a2a + permute) on comm_stream
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(routing_done)
            dispatched_tokens, tokens_per_expert, state = dispatch_tokens(
                x_routed, selected_experts_indices, top_scores,
                num_local_experts, self.num_experts, ep_group,
                score_before_experts=self.score_before_experts,
            )
            dispatch_done = torch.cuda.Event()
            dispatch_done.record(comm_stream)

        self._dispatch_state = state
        self._dispatched_tokens = dispatched_tokens
        self._tokens_per_expert = tokens_per_expert
        self._dispatch_done_event = dispatch_done

        # Load balancing
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                num_tokens_per_expert = torch.histc(
                    selected_experts_indices.float().view(-1),
                    bins=self.num_experts, min=0, max=self.num_experts - 1,
                )
                self.tokens_per_expert.add_(num_tokens_per_expert)

    def wait_dispatch_and_gemm(self) -> torch.Tensor:
        """Wait for dispatch, run Expert GEMM on default stream."""
        torch.cuda.current_stream().wait_event(self._dispatch_done_event)
        routed_output = self.experts.forward(
            self._dispatched_tokens, self._tokens_per_expert
        )
        del self._dispatched_tokens, self._tokens_per_expert, self._dispatch_done_event
        return routed_output

    def combine_async(self, routed_output: torch.Tensor):
        """Phase 3: Start async combine. Returns immediately."""
        from torchtitan.distributed.deepep import combine_tokens

        self._routed_output = combine_tokens(routed_output, self._dispatch_state)
        self._dispatch_state = None

    def shared_forward(self, x_current: torch.Tensor) -> torch.Tensor | None:
        """Phase 3b: SharedExperts on current input (overlaps with combine)."""
        x_current_flat = x_current.view(-1, x_current.shape[-1])
        x_shared = self.shared_norm(x_current_flat)

        if self.shared_experts is None:
            return None

        shared_output = self.shared_experts(x_shared)
        if self.shared_gate is not None:
            shared_output = F.sigmoid(self.shared_gate(x_shared)) * shared_output
        return shared_output

    def sync_and_merge(
        self, shared_output: torch.Tensor | None, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        """Phase 4: Wait for combine, merge routed + shared outputs."""
        from torchtitan.distributed.deepep import sync_combine

        sync_combine()

        routed_output = self._routed_output
        del self._routed_output

        if shared_output is not None:
            output = routed_output + shared_output
        else:
            output = routed_output

        return output.view(shape)

    # ------------------------------------------------------------------ #
    #  Fallback forward — used when shortcut is not yet available         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_current: torch.Tensor,
        x_shortcut: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential fallback (first ScMoE layer or non-overlap path).

        Uses self.experts() WITH hooks (standard DeepEP dispatch/combine).
        """
        from torchtitan.distributed.deepep import sync_combine

        bs, slen, dim = x_current.shape
        x_current_flat = x_current.view(-1, dim)
        x_shortcut_flat = x_shortcut.view(-1, dim)

        x_routed = self.routed_norm(x_shortcut_flat)
        x_shared = self.shared_norm(x_current_flat)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_routed, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Standard path: hooks handle dispatch/combine
        routed_output = self.experts(
            x_routed,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.num_experts,
        )

        # SharedExperts overlaps with async combine
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x_shared)
            if self.shared_gate is not None:
                shared_output = F.sigmoid(self.shared_gate(x_shared)) * shared_output
        else:
            shared_output = None

        sync_combine()

        if shared_output is not None:
            output = routed_output + shared_output
        else:
            output = routed_output

        ScMoEDeepEP._timing_step += 1
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
