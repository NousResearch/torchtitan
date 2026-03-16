# Mamba2 mixer extracted from HF NemotronH (hf_ref.py).
# Stripped of HF transformers dependencies. Keeps both CUDA and torch paths.
#
# This is used instead of FLA's Mamba2 because FLA's torch_forward fallback
# has two bugs vs HF when n_groups > 1:
#   1. Uses repeat() instead of repeat_interleave() for B/C group expansion
#   2. Uses time_step_limit=(0, inf) instead of time_step_min clamping

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# Try to import CUDA kernels (optional, falls back to pure PyTorch)
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    print("Causal conv1d not available")

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
except ImportError:
    print("Mamba_ssm not available")
    selective_state_update = None
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)


class RMSNormGated(nn.Module):
    """Grouped gated RMSNorm matching HF MambaRMSNormGated.

    Normalizes within groups of group_size elements independently.
    group_size=None means normalize over the full dimension (no grouping).
    norm_before_gate=False: norm(x * silu(z))
    norm_before_gate=True:  norm(x) * silu(z)
    """

    def __init__(self, hidden_size, eps=1e-6, norm_before_gate=False, group_size=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size or hidden_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            gate = gate.to(torch.float32)
            if not self.norm_before_gate:
                hidden_states = hidden_states * F.silu(gate)
        # Grouped RMSNorm: reshape to (..., n_groups, group_size), norm per group
        shape = hidden_states.shape
        hidden_states = hidden_states.view(*shape[:-1], -1, self.group_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.view(shape)
        hidden_states = self.weight.float() * hidden_states
        if gate is not None and self.norm_before_gate:
            hidden_states = hidden_states * F.silu(gate)
        return hidden_states.to(input_dtype)


# -- Helpers --


def pad_tensor_by_size(input_tensor, pad_size):
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        return input_tensor.reshape(
            input_tensor.shape[0],
            -1,
            chunk_size,
            input_tensor.shape[2],
            input_tensor.shape[3],
        )


def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


# -- Mamba2 Mixer --


class Mamba2(nn.Module):
    """
    Mamba2 mixer, extracted from HF NemotronHMamba2Mixer.
    Same weight layout as FLA's Mamba2 (in_proj, conv1d, dt_bias, A_log, D, norm, out_proj).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int = 64,
        hidden_size: int = 2048,
        state_size: int = 128,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        chunk_size: int = 256,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit: tuple = None,
        use_bias: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        # expand is NOT used - intermediate_size is num_heads * head_dim
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = num_heads * head_dim
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias
        self.activation = hidden_act
        self.act = F.silu if hidden_act == "silu" else getattr(F, hidden_act, F.silu)
        self.use_mem_eff_path = True

        self.n_groups = n_groups
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=use_bias)

        # dt init
        dt = torch.exp(
            torch.rand(self.num_heads)
            * (math.log(time_step_max) - math.log(time_step_min))
            + math.log(time_step_min)
        ).clamp(min=time_step_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(
            self.intermediate_size,
            eps=norm_eps,
            norm_before_gate=False,
            group_size=self.intermediate_size // self.n_groups,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=use_bias
        )

        if not is_fast_path_available:
            logger.warning(
                "mamba-ssm/causal-conv1d not available, using slow torch path for Mamba2"
            )

    def cuda_kernels_forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = (
            2 * self.intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
            + self.num_heads
        )

        if attention_mask is not None and not torch.all(attention_mask == 1):
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(
                hidden_states.dtype
            )

        projected_states = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = (
            {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
        )

        if self.training:
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
            hidden_states_B_C = causal_conv1d_fn(
                x=hidden_states_B_C.transpose(1, 2),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)[:, :seq_len]
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [
                    self.intermediate_size,
                    groups_time_state_size,
                    groups_time_state_size,
                ],
                dim=-1,
            )
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(
                    hidden_states.dtype
                )
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
            scan_output = scan_output.view(batch_size, seq_len, -1)
            scan_output = self.norm(scan_output, gate)
            out = self.out_proj(scan_output)

        return out

    # fmt: off
    def torch_forward(self, input_states, attention_mask=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # Conv
        ssm_state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)

        A = -torch.exp(self.A_log.float())

        # SSD naive implementation — dt in float32 for numerical stability
        dt = nn.functional.softplus((dt + self.dt_bias).float())
        dt = torch.clamp(dt, self.time_step_min)
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt

        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        L = torch.exp(segment_sum(A))
        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None] * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)

        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

        states_permuted = states.permute(0, 2, 1, 3, 4)
        result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
        new_states = result.permute(0, 2, 1, 3, 4)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states = (C[..., None, :] * states[:, :, None, ...])
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states
    # fmt: on

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if (
            is_fast_path_available
            and "cuda" in self.in_proj.weight.device.type
            and not getattr(self, "_force_torch_path", False)
        ):
            return self.cuda_kernels_forward(hidden_states, attention_mask), None, None
        return self.torch_forward(hidden_states, attention_mask), None, None
