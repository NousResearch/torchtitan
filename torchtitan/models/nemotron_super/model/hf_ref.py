# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_mamba_2_ssm_available,
)

from .configuration_nemotron_h import NemotronHConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.mamba2.modeling_mamba2
if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)

# TODO: Update with correct checkpoint when model is published to HuggingFace Hub
_CHECKPOINT_FOR_DOC = "nvidia/nemotron-h-placeholder"
_CONFIG_FOR_DOC = "NemotronHConfig"


# Helper methods for segment sum computation


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and not torch.all(attention_mask == 1):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

# Adapted from transformers.models.zamba2.modeling_zamba2.Zamba2HybridDynamicCache for the v2 mixer
class NemotronHHybridDynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    is_compileable = False

    def __init__(
        self, config: NemotronHConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: str | None = None
    ):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False
        self.intermediate_size = int(config.mamba_num_heads * config.mamba_head_dim)
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.n_mamba_heads = config.mamba_num_heads
        self.transformer_layers = []
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.conv_states = {}
        self.ssm_states = {}
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba":
                # Only allocate mamba cache for mamba layers
                self.conv_states[i] = torch.zeros(
                    batch_size,
                    self.intermediate_size + 2 * config.n_groups * self.ssm_state_size,
                    self.conv_kernel_size,
                    device=device,
                    dtype=dtype,
                )
                self.ssm_states[i] = torch.zeros(
                    batch_size,
                    self.n_mamba_heads,
                    config.mamba_head_dim,
                    self.ssm_state_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                # For attention and moe layers, use empty tensors
                self.conv_states[i] = torch.tensor([[]] * batch_size, device=device)
                self.ssm_states[i] = torch.tensor([[]] * batch_size, device=device)

            if self.layers_block_type[i] == "attention":
                self.transformer_layers.append(i)
        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        if self.get_seq_length() > 0:
            for layer_idx in range(len(self.key_cache)):
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

                device = self.conv_states[layer_idx].device
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
                device = self.ssm_states[layer_idx].device
                self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, kv_offset

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

class MambaRMSNormGated(torch.nn.Module):
    """
    Gated Root Mean Square Normalization for Mamba layers.

    This normalization variant supports gating, allowing the normalization to be
    modulated by a gating signal. It is specifically designed for use in Mamba blocks
    and supports grouped normalization.

    Args:
        hidden_size (`int`):
            The dimension of the hidden states to normalize.
        group_size (`int`):
            Size of each group for grouped normalization.
        eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the variance for numerical stability.
    """
    def __init__(self, hidden_size, group_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        return rmsnorm_fn(x=hidden_states,
                          weight=self.weight,
                          bias=None,
                          z=gate,
                          eps=self.variance_epsilon,
                          group_size=self.group_size,
                          norm_before_gate=False
        )

# Adapted from transformers.models.zamba2.modeling_zamba2.Zamba2MambaMixer
class NemotronHMamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]
        self.use_mem_eff_path = True

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.num_heads = config.mamba_num_heads
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=config.layer_norm_epsilon, group_size=self.intermediate_size // self.n_groups)
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )


    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[NemotronHHybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.has_previous_state:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask == 1)
            else:
                input_not_masked = True

            if self.use_mem_eff_path and self.training and cache_params is None and input_not_masked:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                if cache_params is not None:
                    hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
                    conv_state = nn.functional.pad(
                        hidden_states_B_C_t, (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_state)
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[NemotronHHybridDynamicCache]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        if cache_params is not None and cache_params.has_previous_state:
            projected_states = self.in_proj(input_states.squeeze(1))
        else:
            if attention_mask is not None and not torch.all(attention_mask==1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                input_states = (input_states * attention_mask[:, :, None]).to(dtype)
            projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            if cache_params.has_previous_state:
                gate = gate.unsqueeze(1)
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                # handle batched generation - states are copied through
                conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
            else:
                hidden_states = hidden_states.transpose(1,2)
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states).transpose(1,2))[:, :seq_len, :]     # [batch, intermediate_size, seq_len]
                if attention_mask is not None and not torch.all(attention_mask==1):
                    dtype = hidden_states.dtype
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_params.has_previous_state:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]


            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            if cache_params is not None and cache_params.has_previous_state:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[NemotronHHybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)


class NemotronHRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for NemotronH.

    NemotronHRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm. It normalizes
    the input using the root mean square of the hidden dimensions, then scales by
    a learned weight parameter.

    Args:
        hidden_size (`int`):
            The dimension of the hidden states to normalize.
        eps (`float`, *optional*, defaults to 1e-6):
            A small value added to the variance for numerical stability.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

class NemotronHBlock(nn.Module):
    """
    A single transformer block in the NemotronH model.

    This block can contain different types of mixers (Mamba, Attention, MLP, or MoE)
    depending on the configuration. Each block applies pre-normalization followed by
    the mixer, then adds a residual connection.

    Args:
        config (`NemotronHConfig`):
            Model configuration specifying the block architecture.
        layer_idx (`int`):
            Index of this block in the model. Used to determine the block type from
            `config.layers_block_type[layer_idx]`.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # M: Mamba2, *: Attention, -: MLP
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config, layer_idx=layer_idx)
        elif self.block_type == "moe":
            self.mixer = NemotronHMoE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Invalid layer pattern {config.hybrid_override_pattern[layer_idx]}")

    def forward(
        self,
        hidden_states,
        past_key_values: Optional[NemotronHHybridDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        if hidden_states.device.type == "cuda":
            stream_context = torch.cuda.stream(torch.cuda.default_stream(hidden_states.device))
        else:
            stream_context = contextlib.nullcontext()

        with stream_context:
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "mamba":
                hidden_states = self.mixer(
                    hidden_states, cache_params=past_key_values, attention_mask=attention_mask
                )
            elif self.block_type == "attention":
                hidden_states, _, _ = self.mixer(
                    hidden_states=hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            elif self.block_type in ["mlp", "moe"]:
                hidden_states = self.mixer(
                    hidden_states
                )
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            hidden_states = residual + hidden_states
            return hidden_states


# Copied from transformers.models.nemotron.modeling_nemotron Nemotron->NemotronH
class NemotronHMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for NemotronH.

    This module implements a standard feed-forward network with one hidden layer,
    applying an activation function between the up and down projections.

    Args:
        config (`NemotronHConfig`):
            Model configuration containing hyperparameters.
        intermediate_size (`int`, *optional*):
            Dimension of the intermediate hidden layer. If not provided, uses `config.intermediate_size`.
        layer_idx (`int`, *optional*):
            Index of the layer in the model. Used for proper cache management.
        is_expert (`bool`, *optional*, defaults to `False`):
            Whether this MLP is used as an expert in a Mixture-of-Experts layer.
    """
    def __init__(self, config, intermediate_size=None, layer_idx: Optional[int] = None, is_expert=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert
        self.hidden_size = config.hidden_size
        input_size = self.hidden_size if not use_latent_size else config.moe_latent_size

        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(input_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, input_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module for NemotronH.

    This module implements a sparse MoE layer with both routed experts and shared experts.
    Tokens are routed to a subset of experts based on learned routing weights, while all
    tokens are processed by shared experts. The architecture supports optional latent
    dimension projection for computational efficiency.

    Args:
        config (`NemotronHConfig`):
            Model configuration containing MoE-specific hyperparameters including:
            - `n_routed_experts`: Number of routed expert MLPs
            - `num_experts_per_tok`: Number of experts each token is routed to
            - `moe_intermediate_size`: Hidden dimension for routed experts
            - `moe_shared_expert_intermediate_size`: Hidden dimension for shared experts
            - `moe_latent_size`: Optional latent dimension for dimensionality reduction
        layer_idx (`int`, *optional*):
            Index of the layer in the model.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx, is_expert=True)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx, is_expert=False
        )

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                # Local empty expert: no-op compute that still marks params as used.
                expert_dtype = expert.down_proj.weight.dtype
                dummy_out = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0).to(expert_dtype))
                final_hidden_states = final_hidden_states + dummy_out

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itself an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)

        hidden_states = hidden_states.view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class NemotronHTopkRouter(nn.Module):
    """
    Top-K routing module for Mixture-of-Experts.

    This router determines which experts should process each token by computing routing
    logits and selecting the top-K experts based on grouped scoring. It implements
    group-based expert selection with score correction for load balancing.

    Args:
        config (`NemotronHConfig`):
            Model configuration containing routing hyperparameters including:
            - `num_experts_per_tok`: Number of experts to route each token to (K)
            - `n_routed_experts`: Total number of available experts
            - `routed_scaling_factor`: Scaling factor applied to routing weights
            - `n_group`: Number of expert groups for grouped routing
            - `topk_group`: Number of groups to select from
            - `norm_topk_prob`: Whether to normalize the top-K routing probabilities
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        """
        Compute expert routing for each token in the input.

        This method performs the following steps:
        1. Compute routing logits using a linear projection
        2. Apply sigmoid activation to get routing scores
        3. Select top-K experts using grouped selection strategy
        4. Gather and optionally normalize the routing weights
        5. Apply scaling factor to final weights

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input hidden states to be routed to experts.

        Returns:
            `tuple` containing:
                - topk_indices (`torch.Tensor` of shape `(batch_size * sequence_length, num_experts_per_tok)`):
                    Indices of the selected experts for each token.
                - topk_weights (`torch.Tensor` of shape `(batch_size * sequence_length, num_experts_per_tok)`):
                    Normalized routing weights for each selected expert, scaled by routed_scaling_factor.
        """
        self._maintain_float32_expert_bias()

        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _maintain_float32_expert_bias(self):
        """
        Ensure e_score_correction_bias stays in float32 for numerical stability.

        This method is called at the start of forward() to revert the bias back to
        float32 if the model was cast to a lower precision dtype (e.g., via model.to(torch.bfloat16)).

        """
        if self.e_score_correction_bias.dtype != torch.float32:
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(torch.float32)

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention forward pass - computes attention weights explicitly."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class NemotronHAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper

    Args:
        config (`NemotronHConfig`):
            Model configuration containing attention parameters like num_attention_heads, num_key_value_heads,
            hidden_size, head_dim, attention_dropout, and attention_bias.
        layer_idx (`int`, *optional*):
            Index of the layer in the model. Required for proper caching during generation. If not provided,
            a warning is emitted and caching may fail.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[NemotronHHybridDynamicCache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Select attention implementation based on config
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if attention_mask is None and q_len > 1:
            mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device=hidden_states.device), diagonal=1)
            attention_mask = mask.view(1, 1, q_len, q_len)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_values


# Copied from transformers.models.mamba2.modeling_mamba2.Mamba2PreTrainedModel
class NemotronHPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NemotronHConfig
    base_model_prefix = "model"
    _no_split_modules = ["NemotronHBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _checkpoint_conversion_mapping = {"backbone": "model"}

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, NemotronHMamba2Mixer):
            if getattr(module.dt_bias, "_is_hf_initialized", False):
                return
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True
        elif isinstance(module, NemotronHTopkRouter):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if getattr(p, "_is_hf_initialized", False):
                    continue
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


@dataclass
# Copied from transformers.models.mamba2.modeling_mamba2.Mamba2Output with MAMBA2->NemotronH,Mamba2->NemotronH
class NemotronHOutput(ModelOutput):
    """
    Class for the NemotronH model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`NemotronHHybridDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[NemotronHHybridDynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.mamba2.modeling_mamba2.MambaCausalLMOutput with Mamba2->NemotronH
class NemotronHCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`NemotronHHybridDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[NemotronHHybridDynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


NEMOTRONH_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NemotronHConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NEMOTRONH_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            If `past_key_values.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        position_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`NemotronHHybridDynamicCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `past_key_values` is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `past_key_values` is passed, `cache_position` should also be passed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


@add_start_docstrings(
    "The bare NemotronH Model transformer outputting raw hidden-states without any specific head on top.",
    NEMOTRONH_START_DOCSTRING,
)
class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(NEMOTRONH_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=NemotronHOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[NemotronHHybridDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, NemotronHOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # From zamba_modeling.py
        if use_cache and past_key_values is None:
            logger.warning_once(
                "NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        hidden_states = inputs_embeds

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # Until HERE

        for layer_idx, mixer_block in enumerate(self.layers):
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            if mixer_block.block_type == "mamba":
                layer_mask = mamba_mask
            elif mixer_block.block_type == "attention":
                layer_mask = causal_mask
            elif mixer_block.block_type in ["mlp", "moe"]:
                layer_mask = None
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, past_key_values, cache_position, layer_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=layer_mask,
                    output_attentions=output_attentions,
                )

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if past_key_values is not None and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)

        return NemotronHOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if cache_position is None:
            target_length = sequence_length
        else:
            target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        if cache_position is not None:
            causal_mask *= (torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)).to(torch.bool)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


def register_nemotron_h_conversion_mapping():
    try:
        from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
        has_conversion_mapping = True
    except ImportError:
        has_conversion_mapping = False

    if not has_conversion_mapping:
        return

    register_checkpoint_conversion_mapping(
        "nemotron_h",
        [
            WeightRenaming("backbone.", "model."),
            WeightRenaming("embedding.weight", "embeddings.weight"),
        ],
        overwrite=True,
    )



@add_start_docstrings(
    """
    The NEMOTRONH Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    """,
    NEMOTRONH_START_DOCSTRING,
)
class NemotronHForCausalLM(NemotronHPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        register_nemotron_h_conversion_mapping()

        # Initialize weights and apply final processing
        self.post_init()

    def _get_key_renaming_mapping(
        self,
        checkpoint_keys: list[str],
        key_mapping: Optional[dict[str, str]] = None,
        loading_base_model_from_task_state_dict: bool = False,
        loading_task_model_from_base_state_dict: bool = False,
    ):
        """Convert backbone.* keys to model.* keys for backward compatibility."""
        if key_mapping is None:
            key_mapping = {"^backbone": "model"}
        else:
            key_mapping = {"^backbone": "model", **key_mapping}

        has_prefix_module = any(s.startswith("backbone") for s in checkpoint_keys)
        if has_prefix_module:
            loading_task_model_from_base_state_dict = False

        return super()._get_key_renaming_mapping(
            checkpoint_keys,
            key_mapping,
            loading_base_model_from_task_state_dict=loading_base_model_from_task_state_dict,
            loading_task_model_from_base_state_dict=loading_task_model_from_base_state_dict,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- has a unique cache type, `NemotronHHybridDynamicCache`

        if past_key_values is None:
            past_key_values = NemotronHHybridDynamicCache(
                self.config, input_ids.shape[0], dtype=self.dtype, device=self.device
            )

        kwargs["logits_to_keep"] = self.config.num_logits_to_keep
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        return model_inputs

    @add_start_docstrings_to_model_forward(NEMOTRONH_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=NemotronHCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[NemotronHHybridDynamicCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[tuple, NemotronHCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        nemotron_h_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = nemotron_h_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + nemotron_h_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NemotronHCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=nemotron_h_outputs.past_key_values,
            hidden_states=nemotron_h_outputs.hidden_states,
            attentions=nemotron_h_outputs.attentions,
        )
