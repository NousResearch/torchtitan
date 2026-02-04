#!/usr/bin/env python3
"""
COMPREHENSIVE TRACE: Log absolutely everything to find exact NaN source.
Trace every tensor through attention computation.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def deep_check(name, tensor, rank, detailed=True):
    """Deep tensor check with all statistics."""
    if tensor is None:
        print(f"[R{rank}] {name}: None")
        return True, {}

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    # Get finite stats
    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        finite_tensor = tensor[finite_mask]
        min_val = finite_tensor.float().min().item()
        max_val = finite_tensor.float().max().item()
        mean_val = finite_tensor.float().mean().item()
        std_val = finite_tensor.float().std().item() if finite_tensor.numel() > 1 else 0
    else:
        min_val = max_val = mean_val = std_val = float("nan")

    status = (
        "OK"
        if not (has_nan or has_inf)
        else f"NaN:{nan_count}"
        if has_nan
        else f"Inf:{inf_count}"
    )

    if detailed or has_nan or has_inf:
        print(
            f"[R{rank}] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f} [{status}]"
        )

    stats = {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
    }
    return not (has_nan or has_inf), stats


def trace_attention_manual(
    q, k, v, rank, global_k=None, global_v=None, local_start=0, seq_len=None
):
    """Manually trace attention computation step by step."""
    print(f"\n[R{rank}] ======== MANUAL ATTENTION TRACE ========")

    # Use global K,V if provided, else local
    k_use = global_k if global_k is not None else k
    v_use = global_v if global_v is not None else v

    deep_check("Q (query)", q, rank)
    deep_check("K (key)", k_use, rank)
    deep_check("V (value)", v_use, rank)

    # Step 1: Q @ K^T
    print(f"\n[R{rank}] --- Step 1: Q @ K^T ---")
    scores = torch.matmul(q.float(), k_use.float().transpose(-2, -1))
    deep_check("Raw scores (Q @ K^T)", scores, rank)

    # Step 2: Scale
    print(f"\n[R{rank}] --- Step 2: Scale ---")
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    print(f"[R{rank}] Scale factor: {scale:.6f}")
    scaled_scores = scores * scale
    deep_check("Scaled scores", scaled_scores, rank)

    # Step 3: Apply causal mask
    if seq_len is not None:
        print(f"\n[R{rank}] --- Step 3: Causal mask ---")
        local_seq_len = q.shape[2]
        kv_len = k_use.shape[2]

        q_pos = torch.arange(
            local_start, local_start + local_seq_len, device=q.device
        ).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        mask = (q_pos >= kv_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, local_seq, kv_len]

        print(f"[R{rank}] Mask shape: {mask.shape}, True count: {mask.sum().item()}")

        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))
        deep_check("Masked scores", masked_scores, rank)
    else:
        masked_scores = scaled_scores

    # Step 4: Softmax
    print(f"\n[R{rank}] --- Step 4: Softmax ---")

    # Check for all -inf rows (would cause NaN)
    all_neg_inf = (masked_scores == float("-inf")).all(dim=-1)
    if all_neg_inf.any():
        print(
            f"[R{rank}] WARNING: {all_neg_inf.sum().item()} rows are all -inf (will cause NaN softmax)"
        )

    # Check row-wise max before softmax
    row_max = masked_scores.max(dim=-1).values
    deep_check("Row max (before softmax)", row_max, rank)

    attn_probs = F.softmax(masked_scores, dim=-1)
    deep_check("Attention probs (after softmax)", attn_probs, rank)

    # Check for rows that became all zeros or NaN
    row_sums = attn_probs.sum(dim=-1)
    deep_check("Attention probs row sums", row_sums, rank)

    # Step 5: Weighted sum
    print(f"\n[R{rank}] --- Step 5: Weighted sum ---")
    output = torch.matmul(attn_probs, v_use.float())
    deep_check("Output (probs @ V)", output, rank)

    print(f"[R{rank}] ======== END MANUAL TRACE ========\n")
    return output


def trace_flex_attention_wrapper(q, k, v, block_mask, compiled_flex, rank):
    """Wrapper to trace flex_attention inputs/outputs."""
    print(f"\n[R{rank}] ======== FLEX_ATTENTION TRACE ========")

    deep_check("Q input to flex_attention", q, rank)
    deep_check("K input to flex_attention", k, rank)
    deep_check("V input to flex_attention", v, rank)

    print(f"[R{rank}] BlockMask shape: {block_mask.shape}")

    # Call flex_attention
    output = compiled_flex(q, k, v, block_mask=block_mask)

    deep_check("Output from flex_attention", output, rank)

    print(f"[R{rank}] ======== END FLEX_ATTENTION TRACE ========\n")
    return output


class TracingAttentionModel(torch.nn.Module):
    """Model with full tracing."""

    def __init__(
        self, hidden_dim, n_heads, head_dim, device, rank, dtype=torch.bfloat16
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

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

        # Initialize with small values
        for p in self.parameters():
            p.data.normal_(0, 0.02)

    def forward(self, x, block_mask, attn_fn, trace=False, local_start=0, seq_len=256):
        B, S, _ = x.shape

        print(f"\n[R{self.rank}] ======== MODEL FORWARD TRACE ========")
        deep_check("Input x", x, self.rank)

        # Q projection
        q = self.q_proj(x)
        deep_check("After q_proj", q, self.rank)

        # K projection
        k = self.k_proj(x)
        deep_check("After k_proj", k, self.rank)

        # V projection
        v = self.v_proj(x)
        deep_check("After v_proj", v, self.rank)

        # Reshape
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        deep_check("Q reshaped", q, self.rank)
        deep_check("K reshaped", k, self.rank)
        deep_check("V reshaped", v, self.rank)

        # Attention
        print(f"\n[R{self.rank}] --- Calling attention ---")
        out = attn_fn(q, k, v, block_mask=block_mask)
        deep_check("Attention output", out, self.rank)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        deep_check("After reshape", out, self.rank)

        out = self.o_proj(out)
        deep_check("After o_proj (final output)", out, self.rank)

        print(f"[R{self.rank}] ======== END MODEL FORWARD TRACE ========\n")
        return out


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'='*80}")
    print(f"[R{rank}] COMPREHENSIVE NaN TRACE")
    print(f"{'='*80}")

    # Configuration
    batch_size = 1
    seq_len = 256
    hidden_dim = 256
    n_heads = 4
    head_dim = 64
    n_steps = 3

    cp_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("cp",)
    )
    local_seq_len = seq_len // world_size
    local_start = rank * local_seq_len

    print(
        f"[R{rank}] Config: seq_len={seq_len}, local_seq_len={local_seq_len}, local_start={local_start}"
    )

    # Create model
    model = TracingAttentionModel(hidden_dim, n_heads, head_dim, device, rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Check initial weights
    print(f"\n[R{rank}] === INITIAL WEIGHTS ===")
    for name, param in model.named_parameters():
        deep_check(f"Init {name}", param.data, rank, detailed=False)

    compiled_flex = torch.compile(flex_attention, dynamic=True)

    # Freqs for CP context
    freqs = torch.randn(seq_len, head_dim // 2, device=device)

    for step in range(1, n_steps + 1):
        print(f"\n{'#'*80}")
        print(f"[R{rank}] #################### STEP {step} ####################")
        print(f"{'#'*80}")

        # Create input
        torch.manual_seed(42 + step * 100)
        x_full = (
            torch.randn(
                batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
            )
            * 0.1
        )
        x_local = x_full[:, local_start : local_start + local_seq_len, :].clone()

        deep_check("Input x_local", x_local, rank)

        # Create block mask
        block_mask = create_cp_block_mask(
            causal_mask,
            B=batch_size,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device_mesh=cp_mesh,
        )
        print(f"[R{rank}] BlockMask created: shape={block_mask.shape}")

        # Buffers for CP
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
                print(f"\n[R{rank}] === INSIDE CONTEXT_PARALLEL ===")

                out = model(
                    x_local,
                    block_mask,
                    compiled_flex,
                    trace=True,
                    local_start=local_start,
                    seq_len=seq_len,
                )

                deep_check("Model output", out, rank)

                # Compute loss
                print(f"\n[R{rank}] === LOSS COMPUTATION ===")
                loss = out.sum()
                deep_check("Loss (sum)", loss.unsqueeze(0), rank)

            print(f"\n[R{rank}] === BACKWARD PASS ===")
            loss.backward()

            # Check gradients
            print(f"\n[R{rank}] === GRADIENTS ===")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    ok, stats = deep_check(
                        f"Grad {name}", param.grad, rank, detailed=False
                    )
                    if not ok:
                        print(f"[R{rank}] *** NaN GRADIENT DETECTED IN {name} ***")

            optimizer.step()

            # Summary
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
                print(f"[R{rank}] *** NaN DETECTED AT STEP {step} ***")
                break

        except Exception as e:
            print(f"[R{rank}] Step {step} ERROR: {e}")
            import traceback

            traceback.print_exc()
            break

    print(f"\n[R{rank}] === TRACE COMPLETE ===\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
