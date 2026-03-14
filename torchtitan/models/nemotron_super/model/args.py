# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import BaseModelArgs

from torchtitan.tools.logging import logger


@dataclass
class NemotronSuperModelArgs(BaseModelArgs):
    """
    Nemotron Super (NemotronH) config.

    Each layer is a (Mamba2, [Attention], MoE) block.
    Layers in attn_layer_idxs also run attention between mamba and MoE.
    Mamba2 is handled by FLA.
    """

    dim: int = 4096
    n_layers: int = 40  # number of (mamba, [attn], moe) blocks
    n_heads: int = 32  # attention heads (only in attn sublayers)
    n_kv_heads: int = 2
    vocab_size: int = 131072
    head_dim: int = 128
    hidden_dim: int = 2688  # dense MLP intermediate (if needed)
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    qk_norm: bool = False
    max_seq_len: int = 262144
    depth_init: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    eos_id: int = 2

    enable_weight_tying: bool = False

    # which layers get an attention sublayer between mamba and moe
    # 120B-A12B: 8 out of 40 layers, roughly every 4-5
    attn_layer_idxs: list[int] = field(
        default_factory=lambda: [3, 7, 11, 16, 21, 26, 31, 35]
    )

    # Mamba2 params (FLA handles the kernel)
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64  # intermediate = mamba_num_heads * mamba_head_dim = 8192
    ssm_state_size: int = 128
    conv_kernel: int = 4
    chunk_size: int = 128
    n_groups: int = 8  # SSM groups (B, C are grouped)
    mamba_expand: int = 2
    mamba_hidden_act: str = "silu"
    use_conv_bias: bool = True
    use_mamba_proj_bias: bool = False
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 0.0001

    # MoE params
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # MTP (multi-token prediction)
    num_nextn_predict_layers: int = 1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_experts = 0
        nparams_moe_other = 0  # router, shared experts, latent proj
        nparams_mamba = 0
        nparams_attn = 0
        nparams_other = 0  # norms, output, etc.

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
            elif "moe.experts." in name:
                nparams_moe_experts += p.numel()
            elif "moe." in name:
                nparams_moe_other += p.numel()
            elif "mamba." in name or "mamba_norm" in name:
                nparams_mamba += p.numel()
            elif "attention." in name or "attn_norm" in name:
                nparams_attn += p.numel()
            else:
                nparams_other += p.numel()

        nparams = (
            nparams_embedding + nparams_moe_experts + nparams_moe_other
            + nparams_mamba + nparams_attn + nparams_other
        )

        # Active expert params: only top_k of num_experts are used per token
        top_k = self.moe_args.top_k
        num_experts = self.moe_args.num_experts
        nparams_moe_active = nparams_moe_other + nparams_moe_experts * top_k // num_experts

        # Active params (everything except inactive experts)
        nparams_active = nparams_mamba + nparams_attn + nparams_other + nparams_moe_active

        # FLOPs per token:
        # Linear layers (fwd + bwd): 6 * active_params
        # Attention S^2 term: only on layers with attention (not all n_layers)
        # Mamba SSM scan: ~2 * mamba_num_heads * mamba_head_dim * ssm_state_size per mamba layer
        n_attn_layers = len(self.attn_layer_idxs)
        n_mamba_layers = self.n_layers  # every layer has mamba

        num_flops_per_token = (
            # matmul FLOPs (fwd + bwd)
            6 * nparams_active
            # attention score computation (QK^T + attn@V), only on attn layers
            + 6 * n_attn_layers * self.n_heads * (2 * self.head_dim) * seq_len
            # mamba SSM scan (B/C state matmuls, approximate)
            + 6 * n_mamba_layers * self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size
        )

        logger.info(
            f"NemotronSuper params: mamba {nparams_mamba:,}, attn {nparams_attn:,}, "
            f"moe {nparams_moe_experts + nparams_moe_other:,} "
            f"(active {nparams_moe_active:,}), "
            f"other {nparams_other + nparams_embedding:,}, "
            f"total {nparams:,}, active {nparams_active + nparams_embedding:,}"
        )

        return nparams, num_flops_per_token

