#!/usr/bin/env python3
"""Benchmark: async offload overlap with compute.

Architecture per layer:
  [Expert MLP] → [Heavy Compute Block (simulates attention)]

The D2H offload of expert activations overlaps with the Heavy Compute.
The H2D reload overlaps with Heavy Compute's backward.

6 runs:
  1. Baseline          — no offload, activations stay on GPU
  2. Recompute only    — torch.utils.checkpoint, no CPU involvement
  3. Sync CPU offload  — D2H blocks GPU, then compute, then H2D blocks
  4. Async CPU offload — D2H overlaps with next compute (our engine)
  5. Async + weights   — async activation offload + weight offload
  6. Async + all       — async activation + weight + gradient offload

Usage:
  python scripts/benchmark_async_overlap.py
"""

import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from torchtitan.distributed.cpu_offload.tensor_offloader import TensorOffloader


# ── Model Components ───────────────────────────────────────────────────

class ExpertBlock(nn.Module):
    """Simulates MoE expert: two FC layers with SiLU."""
    def __init__(self, dim, expert_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, expert_dim, bias=False)
        self.w3 = nn.Linear(dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.silu(self.w1(x)) * self.w3(x))


class HeavyCompute(nn.Module):
    """Simulates attention or other compute that runs AFTER experts.
    This is what we overlap the D2H copy with."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, dim * 4, bias=False)
        self.proj2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.proj2(torch.gelu(self.proj1(h)))


# ── Offload Strategies ─────────────────────────────────────────────────

class AsyncOffloadFunction(torch.autograd.Function):
    """Expert forward with async CPU offload of input.

    Forward: run expert, async D2H of input to CPU (overlaps with next op).
    Backward: async H2D reload, recompute expert for gradients.
    """
    @staticmethod
    def forward(ctx, x, expert, offloader):
        # Run expert
        with torch.no_grad():
            output = expert(x)

        # Async D2H — this will overlap with whatever runs next on default stream
        handle = offloader.offload(x.detach(), release_storage=False)

        ctx.handle = handle
        ctx.offloader = offloader
        ctx.expert = expert
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        # Async H2D reload
        x_reloaded = ctx.offloader.reload(ctx.handle)
        ctx.offloader.sync_reload()

        x_reloaded = x_reloaded.detach().requires_grad_(True)
        with torch.enable_grad():
            output = ctx.expert(x_reloaded)
        output.backward(grad_output)
        return x_reloaded.grad, None, None


class SyncOffloadFunction(torch.autograd.Function):
    """Same as async but blocks on every copy — no overlap."""
    @staticmethod
    def forward(ctx, x, expert, offloader):
        with torch.no_grad():
            output = expert(x)

        # SYNC D2H — blocks GPU until copy finishes
        handle = offloader.offload(x.detach(), release_storage=False)
        offloader.sync_offload()  # ← THIS BLOCKS
        torch.cuda.synchronize()  # ← FULL SYNC

        ctx.handle = handle
        ctx.offloader = offloader
        ctx.expert = expert
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        # SYNC H2D — blocks GPU until copy finishes
        x_reloaded = ctx.offloader.reload(ctx.handle)
        torch.cuda.synchronize()  # ← FULL SYNC

        x_reloaded = x_reloaded.detach().requires_grad_(True)
        with torch.enable_grad():
            output = ctx.expert(x_reloaded)
        output.backward(grad_output)
        return x_reloaded.grad, None, None


# ── Model ──────────────────────────────────────────────────────────────

class BenchmarkModel(nn.Module):
    """N layers of [Expert → HeavyCompute].

    mode:
      "baseline"    — no offload
      "recompute"   — torch.utils.checkpoint on expert
      "sync"        — sync CPU offload (blocking)
      "async"       — async CPU offload (overlapped)
    """
    def __init__(self, dim, expert_dim, n_layers, mode="baseline"):
        super().__init__()
        self.experts = nn.ModuleList([ExpertBlock(dim, expert_dim) for _ in range(n_layers)])
        self.computes = nn.ModuleList([HeavyCompute(dim) for _ in range(n_layers)])
        self.mode = mode
        self.offloader = TensorOffloader(pin_memory=True, use_pool=False) if mode in ("sync", "async") else None

    def forward(self, x):
        for i, (expert, compute) in enumerate(zip(self.experts, self.computes)):
            if self.mode == "baseline":
                x = x + expert(x)
                x = compute(x)

            elif self.mode == "recompute":
                x = x + cp.checkpoint(expert, x, use_reentrant=False, preserve_rng_state=False)
                x = compute(x)

            elif self.mode == "sync":
                expert_out = SyncOffloadFunction.apply(x, expert, self.offloader)
                x = x + expert_out
                x = compute(x)  # compute runs AFTER sync — no overlap

            elif self.mode == "async":
                expert_out = AsyncOffloadFunction.apply(x, expert, self.offloader)
                x = x + expert_out
                x = compute(x)  # D2H overlaps with this compute!

        return x


# ── Benchmark ──────────────────────────────────────────────────────────

def benchmark(mode, dim, expert_dim, n_layers, batch, seq_len, n_warmup=3, n_measure=10):
    torch.manual_seed(42)
    model = BenchmarkModel(dim, expert_dim, n_layers, mode=mode).cuda()
    x = torch.randn(batch, seq_len, dim, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure
    start = time.perf_counter()
    for _ in range(n_measure):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    tokens = batch * seq_len * n_measure
    tps = tokens / elapsed
    ms_per_step = (elapsed / n_measure) * 1000

    del model, optimizer, x
    torch.cuda.empty_cache()

    return peak_mem, tps, ms_per_step, loss.item()


def main():
    # Model config — big enough to stress memory
    dim = 2048
    expert_dim = 4096
    n_layers = 12
    batch = 8
    seq_len = 2048

    print(f"Model: {n_layers} layers, dim={dim}, expert_dim={expert_dim}")
    print(f"Input: batch={batch}, seq={seq_len}, tokens/step={batch*seq_len:,}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    modes = [
        ("baseline",  "No offload — activations stay on GPU"),
        ("recompute", "Recompute only — torch.utils.checkpoint, no CPU"),
        ("sync",      "Sync CPU offload — D2H/H2D block GPU"),
        ("async",     "Async CPU offload — D2H overlaps with HeavyCompute"),
    ]

    results = []
    for mode, desc in modes:
        print(f"Running: {mode} ({desc})...")
        mem, tps, ms, loss = benchmark(mode, dim, expert_dim, n_layers, batch, seq_len)
        results.append((mode, desc, mem, tps, ms, loss))
        print(f"  Memory: {mem:.1f} GiB | TPS: {tps:,.0f} | {ms:.1f} ms/step | loss: {loss:.3f}")
        print()

    # Summary table
    base_mem, base_tps = results[0][2], results[0][3]
    print("=" * 100)
    print(f"{'Config':<12} {'Memory':>10} {'Saved':>10} {'TPS':>10} {'TPS Δ':>10} {'ms/step':>10} {'Loss':>8}")
    print("-" * 100)
    for mode, desc, mem, tps, ms, loss in results:
        saved = base_mem - mem
        tps_delta = ((tps - base_tps) / base_tps) * 100
        print(f"{mode:<12} {mem:>8.1f}GiB {saved:>+8.1f}GiB {tps:>10,.0f} {tps_delta:>+9.1f}% {ms:>8.1f}ms {loss:>8.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
