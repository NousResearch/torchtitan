#!/usr/bin/env python
"""
Minimal debug script to reproduce CP + FlexAttention NaN gradient bug.

This script isolates the CP + FlexAttention interaction to identify
the root cause of NaN gradients when context_parallel_degree >= 2.

Usage:
    # Single GPU test (no CP)
    python debug_cp_flex_attn.py --cp_degree 1

    # Multi-GPU test with CP=2 (requires 2 GPUs)
    torchrun --nproc_per_node=2 debug_cp_flex_attn.py --cp_degree 2
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

# FlexAttention imports
from torch.nn.attention.flex_attention import (
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
)

# CP imports
try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import (
        create_cp_block_mask,
        set_rotate_method,
    )

    HAS_CP_SUPPORT = True
except ImportError:
    HAS_CP_SUPPORT = False
    print("Warning: PyTorch version does not support CP + FlexAttention")


def get_causal_mask_mod():
    """Causal mask modifier for FlexAttention."""

    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return _causal_mask


def get_document_mask_mod(batch: torch.Tensor, eos_id: int):
    """Document mask modifier for FlexAttention (for block_causal)."""
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(b, h, q_idx, kv_idx):
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


class SimpleFlexAttention(nn.Module):
    """Simplified attention module using FlexAttention."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.scale = self.head_dim**-0.5

        # Compile flex_attention for performance
        self._compiled_flex_attn = torch.compile(
            flex_attention, mode="max-autotune-no-cudagraphs"
        )

    def forward(self, x: torch.Tensor, block_mask: BlockMask):
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Use FlexAttention with the block mask
        output = self._compiled_flex_attn(
            q, k, v, block_mask=block_mask, scale=self.scale
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class SimpleTransformer(nn.Module):
    """Simplified transformer for debugging."""

    def __init__(
        self, vocab_size: int, dim: int, n_heads: int, n_layers: int, max_seq_len: int
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Precompute freqs_cis for rotary embeddings (simplified - just positions)
        self.register_buffer(
            "freqs_cis",
            torch.arange(max_seq_len, dtype=torch.float32),
            persistent=False,
        )

        self.layers = nn.ModuleList(
            [SimpleFlexAttention(dim, n_heads) for _ in range(n_layers)]
        )

        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, block_mask: BlockMask):
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = h + layer(h, block_mask)

        h = self.norm(h)
        return self.output(h)


def create_attention_mask(mask_mod, B, H, seq_len, device, cp_mesh=None):
    """Create attention mask, with CP support if cp_mesh provided."""
    if cp_mesh is not None and HAS_CP_SUPPORT:
        print(f"  Creating CP block mask with cp_mesh size={cp_mesh.size()}")
        return create_cp_block_mask(
            mask_mod=mask_mod,
            B=B,
            H=H,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device_mesh=cp_mesh,
        )
    else:
        print(f"  Creating standard block mask")
        _compiled_create_block_mask = torch.compile(create_block_mask)
        return _compiled_create_block_mask(
            mask_mod, B, H, seq_len, seq_len, device=device
        )


def run_debug(args):
    """Run the debug test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"\n{'='*60}")
    print(f"CP + FlexAttention Debug Test")
    print(f"{'='*60}")
    print(f"Rank: {rank}, World Size: {world_size}")
    print(f"CP Degree: {args.cp_degree}")
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Has CP Support: {HAS_CP_SUPPORT}")
    print(f"{'='*60}\n")

    # Initialize distributed if needed
    cp_mesh = None
    if args.cp_degree > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Create device mesh for CP
        cp_mesh = init_device_mesh("cuda", (args.cp_degree,), mesh_dim_names=("cp",))
        print(f"[Rank {rank}] Created CP mesh: {cp_mesh}")

    # Model config
    vocab_size = 1024
    dim = 256
    n_heads = 8
    n_layers = 2
    seq_len = args.seq_len
    batch_size = args.batch_size

    print(f"\nModel Config:")
    print(f"  vocab_size={vocab_size}, dim={dim}, n_heads={n_heads}")
    print(f"  n_layers={n_layers}, seq_len={seq_len}, batch_size={batch_size}")

    # Create model
    torch.manual_seed(42 + rank)
    model = SimpleTransformer(vocab_size, dim, n_heads, n_layers, seq_len).to(device)
    model.train()

    # Create random input
    torch.manual_seed(123)  # Same input on all ranks
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create attention mask
    print(f"\nCreating attention mask...")

    # Test both causal and block_causal
    mask_mods = [get_causal_mask_mod()]
    if args.block_causal:
        # Simulate document boundaries with random EOS tokens
        eos_id = 2
        mask_mods.append(get_document_mask_mod(input_tokens, eos_id))

    combined_mask_mod = and_masks(*mask_mods) if len(mask_mods) > 1 else mask_mods[0]

    block_mask = create_attention_mask(
        combined_mask_mod,
        B=batch_size if args.block_causal else 1,
        H=n_heads,
        seq_len=seq_len,
        device=device,
        cp_mesh=cp_mesh,
    )

    print(f"  Block mask created: {type(block_mask)}")

    # Setup CP context if needed
    cp_context = None
    if cp_mesh is not None and HAS_CP_SUPPORT:
        set_rotate_method(args.rotate_method)
        cp_buffers = [input_tokens, labels, model.freqs_cis]
        cp_seq_dims = [1, 1, 0]
        cp_context = context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=set([input_tokens, labels]),
        )

    # Forward pass
    print(f"\nRunning forward pass...")
    loss_fn = nn.CrossEntropyLoss()

    try:
        if cp_context is not None:
            with cp_context:
                logits = model(input_tokens, block_mask)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        else:
            logits = model(input_tokens, block_mask)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        print(f"  Loss: {loss.item()}")
        print(f"  Loss is NaN: {torch.isnan(loss).item()}")
        print(f"  Loss is Inf: {torch.isinf(loss).item()}")

        # Backward pass
        print(f"\nRunning backward pass...")
        loss.backward()

        # Check gradients
        grad_norm = 0.0
        num_params_with_grad = 0
        num_nan_grads = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                num_params_with_grad += 1
                grad_norm += param.grad.data.norm(2).item() ** 2
                if torch.isnan(param.grad).any():
                    num_nan_grads += 1
                    print(f"  NaN gradient in: {name}")

        grad_norm = grad_norm**0.5

        print(f"\nGradient Stats:")
        print(f"  Total grad norm: {grad_norm}")
        print(f"  Params with grad: {num_params_with_grad}")
        print(f"  Params with NaN grad: {num_nan_grads}")
        print(f"  Grad norm is NaN: {grad_norm != grad_norm}")  # NaN check

        # Final verdict
        print(f"\n{'='*60}")
        if torch.isnan(loss) or (grad_norm != grad_norm):
            print("RESULT: NaN DETECTED - BUG REPRODUCED!")
        else:
            print("RESULT: No NaN detected - Test passed!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Debug CP + FlexAttention NaN issue")
    parser.add_argument(
        "--cp_degree", type=int, default=1, help="Context parallel degree"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--block_causal", action="store_true", help="Use block causal mask"
    )
    parser.add_argument(
        "--rotate_method",
        type=str,
        default="allgather",
        choices=["allgather", "alltoall"],
        help="CP rotation method",
    )

    args = parser.parse_args()
    run_debug(args)


if __name__ == "__main__":
    main()
