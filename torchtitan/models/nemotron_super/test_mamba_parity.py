"""
Test FLA's Mamba2 against the HF NemotronH Mamba2 torch_forward implementation.

This is the ONLY component not shared with the rest of the codebase.
Extracts the HF torch_forward logic from hf_ref.py and runs it standalone.

Usage: python -m torchtitan.models.nemotron_super.test_mamba_parity
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensor_by_size(input_tensor, pad_size):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def hf_mamba2_torch_forward(
    input_states, in_proj, conv1d, dt_bias, A_log, D, norm_weight, out_proj,
    num_heads, head_dim, n_groups, ssm_state_size, conv_kernel_size,
    chunk_size, time_step_min, norm_eps,
):
    """
    Reimplementation of HF NemotronHMamba2Mixer.torch_forward (the naive SSD path).
    Extracted from hf_ref.py lines 478-669.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    intermediate_size = num_heads * head_dim

    # Project
    projected_states = in_proj(input_states)
    d_mlp = (projected_states.shape[-1] - 2 * intermediate_size - 2 * n_groups * ssm_state_size - num_heads) // 2
    _, _, gate, hidden_states_raw, dt = projected_states.split(
        [d_mlp, d_mlp, intermediate_size, intermediate_size + 2 * n_groups * ssm_state_size, num_heads], dim=-1
    )

    # Conv
    conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
    hidden_states_conv = hidden_states_raw
    ssm_state = torch.zeros(
        (batch_size, num_heads, head_dim, ssm_state_size),
        device=hidden_states_conv.device, dtype=dtype
    )
    hidden_states_conv = F.silu(conv1d(hidden_states_conv.transpose(1, 2))[..., :seq_len].transpose(1, 2))
    hidden_states, B, C = torch.split(
        hidden_states_conv,
        [intermediate_size, n_groups * ssm_state_size, n_groups * ssm_state_size],
        dim=-1,
    )

    A = -torch.exp(A_log.float())

    # SSD naive implementation
    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_min)
    hidden_states = hidden_states.reshape(batch_size, seq_len, -1, head_dim).float()
    B = B.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    B = B.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    C = C.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C)]

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
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))

    states_permuted = states.permute(0, 2, 1, 3, 4)
    result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
    new_states = result.permute(0, 2, 1, 3, 4)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    C_times_states = (C[..., None, :] * states[:, :, None, ...])
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

    y = Y_diag + Y_off
    y = y.reshape(batch_size, -1, num_heads, head_dim)
    y = y + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)

    # Gated RMSNorm (norm_before_gate=False means: norm(y) * silu(gate))
    y_flat = y.reshape(-1, y.shape[-1]).float()
    gate_flat = gate.reshape(-1, gate.shape[-1]).float()
    variance = y_flat.pow(2).mean(-1, keepdim=True)
    y_normed = y_flat * torch.rsqrt(variance + norm_eps)
    y_normed = y_normed * norm_weight.float()
    y_normed = y_normed * F.silu(gate_flat)
    scan_output = y_normed.reshape(batch_size, seq_len, -1)

    output = out_proj(scan_output.to(dtype))
    return output


def main():
    device = "cuda"
    torch.manual_seed(42)

    # Config that satisfies: expand * hidden_size == num_heads * head_dim
    hidden_size = 64
    expand = 2
    num_heads = 8
    head_dim = 16  # 8 * 16 = 128 = 2 * 64
    n_groups = 2
    ssm_state_size = 16
    conv_kernel = 4
    chunk_size = 32
    time_step_min = 0.001
    time_step_max = 0.1
    norm_eps = 1e-5

    intermediate_size = num_heads * head_dim  # 128
    assert intermediate_size == expand * hidden_size

    print(f"Config: hidden={hidden_size}, intermediate={intermediate_size}, "
          f"heads={num_heads}, head_dim={head_dim}, groups={n_groups}")

    # Create FLA's Mamba2
    from fla.models.mamba2.modeling_mamba2 import Mamba2

    fla_mamba = Mamba2(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        state_size=ssm_state_size,
        expand=expand,
        n_groups=n_groups,
        conv_kernel=conv_kernel,
        use_conv_bias=True,
        hidden_act="silu",
        chunk_size=chunk_size,
        time_step_min=time_step_min,
        time_step_max=time_step_max,
        use_bias=False,
        norm_eps=norm_eps,
    ).to(device).float()

    # Extract weights
    in_proj = fla_mamba.in_proj
    conv1d = fla_mamba.conv1d
    dt_bias = fla_mamba.dt_bias
    A_log = fla_mamba.A_log
    D = fla_mamba.D
    norm_weight = fla_mamba.norm.weight
    out_proj = fla_mamba.out_proj

    # Input
    x = torch.randn(1, 16, hidden_size, device=device)

    with torch.no_grad():
        # FLA forward
        fla_out, _, _ = fla_mamba(x)

        # HF manual forward
        hf_out = hf_mamba2_torch_forward(
            x, in_proj, conv1d, dt_bias, A_log, D, norm_weight, out_proj,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            ssm_state_size=ssm_state_size,
            conv_kernel_size=conv_kernel,
            chunk_size=chunk_size,
            time_step_min=time_step_min,
            norm_eps=norm_eps,
        )

    diff = (fla_out - hf_out).abs().max().item()
    rel = diff / (fla_out.abs().max().item() + 1e-10)

    print(f"\nFLA output std:    {fla_out.std().item():.6f}")
    print(f"HF output std:     {hf_out.std().item():.6f}")
    print(f"Max abs diff:      {diff:.2e}")
    print(f"Max rel diff:      {rel:.2e}")
    print(f"Mean abs diff:     {(fla_out - hf_out).abs().mean().item():.2e}")

    if rel < 1e-3:
        print("\nPASS - FLA Mamba2 matches HF implementation")
    elif rel < 0.05:
        print(f"\nWARN - small numerical diff ({rel:.1%}), probably float precision")
    else:
        print(f"\nFAIL - {rel:.1%} relative error, something is wrong")

        # Debug: print intermediate values
        print("\n--- Debug ---")
        print(f"FLA out[0,0,:8]:  {fla_out[0, 0, :8]}")
        print(f"HF out[0,0,:8]:   {hf_out[0, 0, :8]}")
        print(f"FLA out[0,-1,:8]: {fla_out[0, -1, :8]}")
        print(f"HF out[0,-1,:8]:  {hf_out[0, -1, :8]}")


if __name__ == "__main__":
    main()
