#!/usr/bin/env python3
"""
Test with the ACTUAL FlexAttentionWrapper from torchtitan to reproduce NaN.
"""

import os
import sys

sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)

# Import the actual wrapper
from torchtitan.models.attention import FlexAttentionWrapper


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def deep_check(name, tensor, rank):
    if tensor is None:
        print(f"[R{rank}] {name}: None")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        min_val = tensor[finite_mask].float().min().item()
        max_val = tensor[finite_mask].float().max().item()
        mean_val = tensor[finite_mask].float().mean().item()
    else:
        min_val = max_val = mean_val = float("nan")

    status = (
        "OK"
        if not (has_nan or has_inf)
        else f"NaN:{nan_count}"
        if has_nan
        else f"Inf:{inf_count}"
    )
    print(
        f"[R{rank}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f} [{status}]"
    )

    return not (has_nan or has_inf)


class DeepSeekStyleModel(torch.nn.Module):
    """Model that mimics DeepSeek attention structure."""

    def __init__(
        self,
        hidden_dim,
        n_heads,
        qk_head_dim,
        v_head_dim,
        device,
        rank,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.rank = rank

        # Projections
        self.wq = torch.nn.Linear(
            hidden_dim, n_heads * qk_head_dim, bias=False, device=device, dtype=dtype
        )
        self.wk = torch.nn.Linear(
            hidden_dim, n_heads * qk_head_dim, bias=False, device=device, dtype=dtype
        )
        self.wv = torch.nn.Linear(
            hidden_dim, n_heads * v_head_dim, bias=False, device=device, dtype=dtype
        )
        self.wo = torch.nn.Linear(
            n_heads * v_head_dim, hidden_dim, bias=False, device=device, dtype=dtype
        )

        # Use the ACTUAL FlexAttentionWrapper
        self.inner_attention = FlexAttentionWrapper()

        # Custom softmax scale (like DeepSeek)
        self.softmax_scale = 1.0 / (qk_head_dim**0.5)

        # Initialize weights
        for p in self.parameters():
            p.data.normal_(0, 0.02)

    def forward(self, x, block_mask):
        B, S, _ = x.shape

        print(f"\n[R{self.rank}] === FORWARD START ===")
        deep_check("Input x", x, self.rank)

        q = self.wq(x)
        deep_check("After wq", q, self.rank)

        k = self.wk(x)
        deep_check("After wk", k, self.rank)

        v = self.wv(x)
        deep_check("After wv", v, self.rank)

        q = q.view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.v_head_dim).transpose(1, 2)

        deep_check("Q reshaped", q, self.rank)
        deep_check("K reshaped", k, self.rank)
        deep_check("V reshaped", v, self.rank)

        print(
            f"[R{self.rank}] Calling FlexAttentionWrapper with scale={self.softmax_scale:.6f}"
        )

        # Call the ACTUAL FlexAttentionWrapper
        output = self.inner_attention(
            q, k, v, block_mask=block_mask, scale=self.softmax_scale
        )
        deep_check("Attention output", output, self.rank)

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        deep_check("After reshape", output, self.rank)

        output = self.wo(output)
        deep_check("After wo (final)", output, self.rank)

        print(f"[R{self.rank}] === FORWARD END ===\n")
        return output


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*80}")
    print(f"[R{rank}] Testing with ACTUAL FlexAttentionWrapper")
    print(f"{'='*80}")

    # Configuration - mimic DeepSeek
    batch_size = 1
    seq_len = 256
    hidden_dim = 256
    n_heads = 4
    qk_head_dim = 192  # DeepSeek style: 128 nope + 64 rope
    v_head_dim = 128
    n_steps = 5

    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

    print(
        f"[R{rank}] Config: seq_len={seq_len}, local_seq_len={local_seq_len}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}"
    )

    # Create model
    model = DeepSeekStyleModel(
        hidden_dim, n_heads, qk_head_dim, v_head_dim, device, rank
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Buffers for CP
    freqs = torch.randn(seq_len, qk_head_dim // 2, device=device)

    for step in range(1, n_steps + 1):
        print(f"\n{'#'*80}")
        print(f"[R{rank}] #################### STEP {step} ####################")
        print(f"{'#'*80}")

        torch.manual_seed(42 + step * 100)
        x_full = (
            torch.randn(
                batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
            )
            * 0.1
        )
        x_local = x_full[:, local_start : local_start + local_seq_len, :].clone()

        block_mask = create_cp_block_mask(
            causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device_mesh=cp_mesh,
        )
        print(f"[R{rank}] BlockMask created: shape={block_mask.shape}")

        dummy_input = torch.zeros(batch_size, local_seq_len, device=device)
        dummy_label = torch.zeros(batch_size, local_seq_len, device=device)
        freqs_copy = freqs.clone()

        cp_buffers = [dummy_input, dummy_label, freqs_copy]
        cp_seq_dims = [1, 1, 0]

        optimizer.zero_grad()

        try:
            with context_parallel(
                cp_mesh,
                buffers=cp_buffers,
                buffer_seq_dims=cp_seq_dims,
                no_restore_buffers={dummy_input, dummy_label},
            ):
                out = model(x_local, block_mask)
                loss = out.sum()

            print(f"\n[R{rank}] === BACKWARD ===")
            loss.backward()

            print(f"\n[R{rank}] === GRADIENTS ===")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    ok = deep_check(f"Grad {name}", param.grad, rank)
                    if not ok:
                        print(f"[R{rank}] *** NaN GRADIENT IN {name} ***")

            optimizer.step()

            loss_val = loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            loss_ok = not torch.isnan(loss)
            grad_ok = not torch.isnan(grad_norm)

            print(f"\n[R{rank}] === STEP {step} SUMMARY ===")
            print(f"[R{rank}] Loss: {loss_val:.4f} ({'OK' if loss_ok else 'NaN!'})")
            print(
                f"[R{rank}] Grad norm: {grad_norm.item():.4f} ({'OK' if grad_ok else 'NaN!'})"
            )

            if not loss_ok or not grad_ok:
                print(f"[R{rank}] *** NaN AT STEP {step} ***")
                break

        except Exception as e:
            print(f"[R{rank}] Step {step} ERROR: {e}")
            import traceback

            traceback.print_exc()
            break

    print(f"\n[R{rank}] === TEST COMPLETE ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
