#!/usr/bin/env python3
"""
Multi-step test to reproduce NaN with dynamic=True on step 2+.
Key insight: step 1 works, step 2 crashes or produces NaN.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def check_tensor(name, tensor, rank):
    if tensor is None:
        return "None"
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    status = "NaN!" if has_nan else "Inf!" if has_inf else "OK"
    return status


def precompute_freqs_cis(dim: int, seq_len: int, device: torch.device):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).float()


class SimpleAttentionModel(torch.nn.Module):
    """Simple model with linear projections + flex_attention."""

    def __init__(self, hidden_dim, n_heads, head_dim, device, dtype=torch.bfloat16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.q_proj = torch.nn.Linear(
            hidden_dim, n_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.k_proj = torch.nn.Linear(
            hidden_dim, n_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.v_proj = torch.nn.Linear(
            hidden_dim, n_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.o_proj = torch.nn.Linear(
            n_heads * head_dim, hidden_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x, block_mask, attn_fn):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = attn_fn(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*80}")
    print(f"[Rank {rank}] Multi-Step Test: CP + FlexAttention + dynamic=True")
    print(f"{'='*80}")

    # Configuration
    batch_size = 1
    seq_len = 256
    hidden_dim = 256
    n_heads = 4
    head_dim = 64
    n_steps = 5

    # Create CP mesh
    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    local_seq_len = seq_len // world_size

    print(
        f"[Rank {rank}] seq_len={seq_len}, local_seq_len={local_seq_len}, n_steps={n_steps}"
    )

    # Create model and optimizer
    model = SimpleAttentionModel(hidden_dim, n_heads, head_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create compiled flex_attention with dynamic=True
    compiled_flex = torch.compile(flex_attention, dynamic=True)

    # Create buffers for context_parallel
    freqs_cis_full = precompute_freqs_cis(head_dim, seq_len, device)

    print(f"[Rank {rank}] Model created, starting training loop...")

    for step in range(1, n_steps + 1):
        # Create new input data each step (simulates data loading)
        torch.manual_seed(42 + step * 100)
        x_full = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
        )
        x_local = x_full[
            :, rank * local_seq_len : (rank + 1) * local_seq_len, :
        ].clone()

        # Create block mask for this step
        block_mask = create_cp_block_mask(
            causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device_mesh=cp_mesh,
        )

        # Prepare buffers
        dummy_input = torch.zeros(batch_size, local_seq_len, device=device)
        dummy_label = torch.zeros(batch_size, local_seq_len, device=device)
        freqs_cis = freqs_cis_full.clone()

        cp_buffers = [dummy_input, dummy_label, freqs_cis]
        cp_seq_dims = [1, 1, 0]

        optimizer.zero_grad()

        try:
            with context_parallel(
                cp_mesh,
                buffers=cp_buffers,
                buffer_seq_dims=cp_seq_dims,
                no_restore_buffers={dummy_input, dummy_label},
            ):
                out = model(x_local, block_mask, compiled_flex)
                loss = out.sum()

            loss.backward()
            optimizer.step()

            # Check for NaN
            loss_val = loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            fwd_status = check_tensor("output", out, rank)
            loss_status = "NaN!" if torch.isnan(loss) else "OK"
            grad_status = "NaN!" if torch.isnan(grad_norm) else "OK"

            print(
                f"[Rank {rank}] Step {step}: loss={loss_val:.4f} ({loss_status}), "
                f"grad_norm={grad_norm.item():.4f} ({grad_status}), out={fwd_status}"
            )

            if loss_status == "NaN!" or grad_status == "NaN!":
                print(f"[Rank {rank}] *** NaN DETECTED AT STEP {step} ***")

        except Exception as e:
            print(f"[Rank {rank}] Step {step} ERROR: {e}")
            import traceback

            traceback.print_exc()
            break

    print(f"\n[Rank {rank}] === Multi-Step Test Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
