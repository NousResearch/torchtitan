#!/usr/bin/env python3
"""
Test with larger head dimensions like DeepSeek (head_dim=192).
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import (
    context_parallel,
    create_cp_block_mask,
)
from torch.nn.attention.flex_attention import flex_attention


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
    return "NaN!" if has_nan else "Inf!" if has_inf else "OK"


def precompute_freqs_cis(dim: int, seq_len: int, device: torch.device):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).float()


class DeepSeekStyleAttention(torch.nn.Module):
    """DeepSeek-style attention with qk_nope_head_dim + qk_rope_head_dim."""

    def __init__(
        self,
        hidden_dim,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        device,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.qk_head_dim = (
            qk_nope_head_dim + qk_rope_head_dim
        )  # 128 + 64 = 192 for DeepSeek
        self.v_head_dim = v_head_dim

        # Q and K projections produce larger tensors
        self.q_proj = torch.nn.Linear(
            hidden_dim,
            n_heads * self.qk_head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.k_proj = torch.nn.Linear(
            hidden_dim,
            n_heads * self.qk_head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            hidden_dim, n_heads * v_head_dim, bias=False, device=device, dtype=dtype
        )
        self.o_proj = torch.nn.Linear(
            n_heads * v_head_dim, hidden_dim, bias=False, device=device, dtype=dtype
        )

    def forward(self, x, block_mask, attn_fn):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.v_head_dim).transpose(1, 2)

        # Note: flex_attention requires Q,K to have same head_dim as V for correctness
        # DeepSeek handles this differently, but for testing we use the Q,K head_dim
        out = attn_fn(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


def test_config(
    name,
    hidden_dim,
    n_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    seq_len,
    n_steps,
    rank,
    device,
    cp_mesh,
):
    """Test a specific configuration."""
    print(f"\n[Rank {rank}] === TEST: {name} ===")
    print(
        f"[Rank {rank}] qk_head_dim={qk_nope_head_dim+qk_rope_head_dim}, v_head_dim={v_head_dim}, seq_len={seq_len}"
    )

    world_size = dist.get_world_size()
    batch_size = 1
    local_seq_len = seq_len // world_size

    # Create model
    model = DeepSeekStyleAttention(
        hidden_dim, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    compiled_flex = torch.compile(flex_attention, dynamic=True)

    freqs_cis_full = precompute_freqs_cis(
        qk_nope_head_dim + qk_rope_head_dim, seq_len, device
    )

    results = []
    for step in range(1, n_steps + 1):
        torch.manual_seed(42 + step * 100)
        x_full = (
            torch.randn(
                batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
            )
            * 0.1
        )
        x_local = x_full[
            :, rank * local_seq_len : (rank + 1) * local_seq_len, :
        ].clone()

        block_mask = create_cp_block_mask(
            causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device_mesh=cp_mesh,
        )

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

            loss_val = loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )

            is_nan = torch.isnan(loss) or torch.isnan(grad_norm)
            status = "NaN!" if is_nan else "OK"
            results.append(status)

            print(
                f"[Rank {rank}] Step {step}: loss={loss_val:.4f}, grad_norm={grad_norm.item():.4f} {status}"
            )

        except Exception as e:
            print(f"[Rank {rank}] Step {step} ERROR: {e}")
            results.append("ERROR")
            break

    return results


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*80}")
    print(f"[Rank {rank}] Testing Large Head Dimensions (DeepSeek-style)")
    print(f"{'='*80}")

    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )

    n_steps = 5
    all_results = {}

    # Test 1: Small head dim (baseline, known working)
    all_results["small_head_64"] = test_config(
        "Small head_dim=64",
        hidden_dim=256,
        n_heads=4,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        seq_len=256,
        n_steps=n_steps,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 2: Medium head dim
    all_results["medium_head_128"] = test_config(
        "Medium head_dim=128",
        hidden_dim=512,
        n_heads=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        v_head_dim=128,
        seq_len=256,
        n_steps=n_steps,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 3: DeepSeek-style head dim (192 = 128 nope + 64 rope)
    all_results["deepseek_head_192"] = test_config(
        "DeepSeek head_dim=192 (128+64)",
        hidden_dim=512,
        n_heads=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        seq_len=256,
        n_steps=n_steps,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 4: Larger sequence length
    all_results["large_seq_2048"] = test_config(
        "Large seq_len=2048, head_dim=64",
        hidden_dim=256,
        n_heads=4,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        seq_len=2048,
        n_steps=n_steps,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Test 5: DeepSeek-style with larger sequence
    all_results["deepseek_large_seq"] = test_config(
        "DeepSeek head_dim=192, seq_len=1024",
        hidden_dim=512,
        n_heads=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        seq_len=1024,
        n_steps=n_steps,
        rank=rank,
        device=device,
        cp_mesh=cp_mesh,
    )

    # Summary
    print(f"\n{'='*80}")
    print(f"[Rank {rank}] === RESULTS SUMMARY ===")
    print(f"{'='*80}")
    for test_name, results in all_results.items():
        all_ok = all(r == "OK" for r in results)
        status = "PASS" if all_ok else f"FAIL ({results})"
        print(f"[Rank {rank}] {test_name}: {status}")

    print(f"\n[Rank {rank}] === Tests Complete ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
