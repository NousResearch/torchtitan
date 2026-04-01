# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh


def balanced_shard_sequence(
    x: torch.Tensor, world_size: int, rank: int, seq_dim: int = 1
) -> torch.Tensor:
    S = x.shape[seq_dim]
    half = S // (2 * world_size)
    head = x.narrow(seq_dim, rank * half, half)
    tail = x.narrow(seq_dim, S - (rank + 1) * half, half)
    return torch.cat([head, tail], dim=seq_dim).contiguous()


def stripe_sequence(
    x: torch.Tensor, world_size: int, rank: int, seq_dim: int = 1
) -> torch.Tensor:
    idx = torch.arange(rank, x.shape[seq_dim], world_size, device=x.device)
    return x.index_select(seq_dim, idx).contiguous()


def _allgather_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return funcol.all_gather_tensor(x, gather_dim=1, group=group)


def _fa4_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute import flash_attn_func
    return flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
        return_lse=True,
    )


def _merge_attn_outputs(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # lse is (B, H, S) from FA4 — permute to (B, S, H, 1) to broadcast with out (B, S, H, D)
    lse1 = lse1.permute(0, 2, 1).unsqueeze(-1).float()
    lse2 = lse2.permute(0, 2, 1).unsqueeze(-1).float()
    out1 = out1.float()
    out2 = out2.float()
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - max_lse)
    w2 = torch.exp(lse2 - max_lse)
    out = (out1 * w1 + out2 * w2) / (w1 + w2)
    lse = (max_lse + torch.log(w1 + w2)).squeeze(-1).permute(0, 2, 1)
    return out, lse


def _fa4_cp_round_robin(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> torch.Tensor:
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)
    S_local = q.shape[1]
    S_half = S_local // 2
    assert S_half > 0, f"S_local={S_local} must be >= 2 for round_robin CP"

    if world_size == 1:
        out, _ = _fa4_call(q, k, v, softmax_scale, causal, deterministic)
        return out

    k_all = _allgather_seq(k, cp_group)
    v_all = _allgather_seq(v, cp_group)

    out_acc: torch.Tensor | None = None
    lse_acc: torch.Tensor | None = None

    for i in range(world_size):
        source_rank = (rank - i) % world_size
        ks = k_all[:, source_rank * S_local : (source_rank + 1) * S_local]
        vs = v_all[:, source_rank * S_local : (source_rank + 1) * S_local]

        partial = False
        if i == 0:
            q_use, k_use, v_use, use_causal = q, ks, vs, causal
        elif not causal:
            q_use, k_use, v_use, use_causal = q, ks, vs, False
        elif source_rank < rank:
            # past rank: K tail is future to all local Q, use head chunk only
            q_use, k_use, v_use, use_causal = q, ks[:, :S_half], vs[:, :S_half], False
        else:
            # future rank: Q head precedes all K, use tail chunk only
            q_use, k_use, v_use, use_causal = q[:, S_half:], ks, vs, False
            partial = True

        out_i, lse_i = _fa4_call(q_use, k_use, v_use, softmax_scale, use_causal, deterministic)

        if out_acc is None:
            out_acc = out_i.float()
            lse_acc = lse_i
        elif partial:
            out_tail, lse_tail = _merge_attn_outputs(
                out_acc[:, S_half:], lse_acc[:, :, S_half:], out_i, lse_i,
            )
            out_acc = torch.cat([out_acc[:, :S_half], out_tail], dim=1)
            lse_acc = torch.cat([lse_acc[:, :, :S_half], lse_tail], dim=2)
        else:
            out_acc, lse_acc = _merge_attn_outputs(out_acc, lse_acc, out_i, lse_i)

    assert out_acc is not None
    return out_acc


def _fa4_cp_striped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> torch.Tensor:
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)
    S_local = q.shape[1]
    assert S_local > 1, f"S_local={S_local} must be >= 2 for striped CP"

    if world_size == 1:
        out, _ = _fa4_call(q, k, v, softmax_scale, causal, deterministic)
        return out

    k_all = _allgather_seq(k, cp_group)
    v_all = _allgather_seq(v, cp_group)

    out_acc: torch.Tensor | None = None
    lse_acc: torch.Tensor | None = None

    for i in range(world_size):
        source_rank = (rank - i) % world_size
        ks = k_all[:, source_rank * S_local : (source_rank + 1) * S_local]
        vs = v_all[:, source_rank * S_local : (source_rank + 1) * S_local]

        if not causal or i == 0 or source_rank < rank:
            out_i, lse_i = _fa4_call(q, ks, vs, softmax_scale, causal, deterministic)
            if out_acc is None:
                out_acc = out_i.float()
                lse_acc = lse_i
            else:
                out_acc, lse_acc = _merge_attn_outputs(out_acc, lse_acc, out_i, lse_i)
        else:
            # future rank: Q[j] -> K[0..j-1], equivalent to causal=True on Q[1:]
            # Q[0] (global pos=rank) is always before all K tokens from source>rank
            out_i, lse_i = _fa4_call(q[:, 1:], ks, vs, softmax_scale, True, deterministic)
            out_tail, lse_tail = _merge_attn_outputs(
                out_acc[:, 1:], lse_acc[:, :, 1:], out_i, lse_i,
            )
            out_acc = torch.cat([out_acc[:, :1], out_tail], dim=1)
            lse_acc = torch.cat([lse_acc[:, :, :1], lse_tail], dim=2)

    assert out_acc is not None
    return out_acc


def _fa4_cp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    distribution: str,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> torch.Tensor:
    if distribution == "round_robin":
        return _fa4_cp_round_robin(q, k, v, cp_group, softmax_scale, causal, deterministic)
    elif distribution == "striped":
        return _fa4_cp_striped(q, k, v, cp_group, softmax_scale, causal, deterministic)
    else:
        raise ValueError(
            f"Unknown CP distribution: {distribution!r}. Use 'round_robin' or 'striped'."
        )


class FA4ContextParallelWrapper(nn.Module):

    def __init__(self, cp_mesh: DeviceMesh, distribution: str = "round_robin"):
        super().__init__()
        self.cp_mesh = cp_mesh
        self.distribution = distribution
        self._cp_group: dist.ProcessGroup | None = None

    @property
    def cp_group(self) -> dist.ProcessGroup:
        if self._cp_group is None:
            self._cp_group = self.cp_mesh.get_group()
        return self._cp_group

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_casual: bool = True,
        deterministic: bool = False,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = _fa4_cp_attn(
            q, k, v,
            self.cp_group,
            self.distribution,
            scale,
            is_casual,
            deterministic,
        )
        return out.transpose(1, 2).to(q.dtype)


def prepare_fa4_cp_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict,
    cp_mesh: DeviceMesh,
    device: torch.device,
    distribution: str = "round_robin",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    rank = cp_mesh.get_local_rank()
    world_size = cp_mesh.size()
    seq_len = inputs.shape[1]

    positions = extra_kwargs.get("positions", None)
    if positions is None:
        positions = torch.arange(seq_len, dtype=torch.int32, device=device).expand(
            inputs.shape
        )

    if distribution == "round_robin":
        inputs    = balanced_shard_sequence(inputs,    world_size, rank, seq_dim=1)
        labels    = balanced_shard_sequence(labels,    world_size, rank, seq_dim=1)
        positions = balanced_shard_sequence(positions, world_size, rank, seq_dim=1)
    else:
        inputs    = stripe_sequence(inputs,    world_size, rank, seq_dim=1)
        labels    = stripe_sequence(labels,    world_size, rank, seq_dim=1)
        positions = stripe_sequence(positions, world_size, rank, seq_dim=1)

    extra_kwargs["positions"] = positions
    return inputs, labels, extra_kwargs


def apply_fa4_cp_to_attention_module(
    attention_modules: list[nn.Module],
    cp_mesh: DeviceMesh,
    distribution: str = "round_robin",
) -> None:
    from torchtitan.models.attention import FlashAttention4Wrapper
    from torchtitan.tools.logging import logger

    wrapper = FA4ContextParallelWrapper(cp_mesh, distribution)

    replaced = 0
    for i, mod in enumerate(attention_modules):
        if isinstance(mod, FlashAttention4Wrapper):
            attention_modules[i] = wrapper
            replaced += 1

    logger.info(f"Applied FA4 CP ({distribution}) to {replaced} attention modules")
