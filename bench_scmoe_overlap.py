"""
Benchmark: ScMoE communication-computation overlap demo.

Simulates slow EP all-to-all (dispatch/combine) using torch.cuda._sleep()
on a comm_stream, and shows ScMoE overlapping shared_experts with it.

Standard MoE:  [Dispatch] → [Expert GEMM] → [Combine] → [SharedExperts]
ScMoE:         [SharedExperts]  [Expert GEMM]
               [Dispatch......] [Combine......]
               ↑ overlaps        ↑ overlaps

Usage:
    python bench_scmoe_overlap.py [--delay_ms 50] [--dim 2048] [--hidden 1536]
"""

import argparse
import time

import torch
import torch.nn as nn

from torchtitan.models.moe.moe import FeedForward, MoEArgs, ScMoEConfig
from torchtitan.models.moe.scmoe import ScMoE


def benchmark_standard_moe(
    scmoe: ScMoE,
    x: torch.Tensor,
    delay_cycles: int,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Standard MoE: all sequential on default stream, no overlap."""
    bs, slen, dim = x.shape
    x_flat = x.view(-1, dim)

    # Warmup
    for _ in range(warmup):
        x_normed = scmoe.shared_norm(x_flat)
        # Simulate dispatch (sequential, blocks default stream)
        torch.cuda._sleep(delay_cycles)
        routed = scmoe._routed_forward(scmoe.routed_norm(x_flat), bs, slen, dim)
        # Simulate combine (sequential)
        torch.cuda._sleep(delay_cycles)
        shared = scmoe._shared_forward(x_normed)
        out = routed + shared if shared is not None else routed
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        x_normed = scmoe.shared_norm(x_flat)
        torch.cuda._sleep(delay_cycles)
        routed = scmoe._routed_forward(scmoe.routed_norm(x_flat), bs, slen, dim)
        torch.cuda._sleep(delay_cycles)
        shared = scmoe._shared_forward(x_normed)
        out = routed + shared if shared is not None else routed
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms per iter


def benchmark_scmoe_overlap(
    scmoe: ScMoE,
    x_current: torch.Tensor,
    x_shortcut: torch.Tensor,
    delay_cycles: int,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """ScMoE: shared_experts overlaps with dispatch/combine on comm_stream."""
    bs, slen, dim = x_current.shape
    comm_stream = torch.cuda.Stream()

    def run_once():
        x_cur = x_current.view(-1, dim)
        x_sc = x_shortcut.view(-1, dim)

        x_routed = scmoe.routed_norm(x_sc)
        x_shared = scmoe.shared_norm(x_cur)

        # Record readiness on default stream
        ready = torch.cuda.Event()
        ready.record()

        # --- Comm stream: simulate dispatch (async) ---
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready)
            torch.cuda._sleep(delay_cycles)
            dispatch_done = torch.cuda.Event()
            dispatch_done.record(comm_stream)

        # --- Default stream: shared_experts (overlaps with dispatch) ---
        shared_out = scmoe._shared_forward(x_shared)

        # Wait for dispatch to complete before Expert GEMM
        torch.cuda.current_stream().wait_event(dispatch_done)

        # --- Default stream: Expert GEMM ---
        routed_out = scmoe._routed_forward(x_routed, bs, slen, dim)

        # --- Comm stream: simulate combine (async) ---
        gemm_done = torch.cuda.Event()
        gemm_done.record()
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(gemm_done)
            torch.cuda._sleep(delay_cycles)
            combine_done = torch.cuda.Event()
            combine_done.record(comm_stream)

        # Wait for combine
        torch.cuda.current_stream().wait_event(combine_done)

        if shared_out is not None:
            out = shared_out + routed_out
        else:
            out = routed_out
        return out

    # Warmup
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms per iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay_ms", type=float, default=50, help="Simulated comm delay in ms")
    parser.add_argument("--dim", type=int, default=2048, help="Model dimension")
    parser.add_argument("--hidden", type=int, default=1536, help="Expert hidden dim")
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_shared_experts", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--slen", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Convert ms to CUDA clock cycles (~1.3 GHz on modern GPUs)
    # torch.cuda._sleep takes cycles, ~1 cycle per ns at ~1GHz
    # Empirically calibrate: 1M cycles ≈ 1ms on most GPUs
    delay_cycles = int(args.delay_ms * 1_000_000)

    print(f"Config: dim={args.dim}, hidden={args.hidden}, experts={args.num_experts}, "
          f"shared={args.num_shared_experts}, top_k={args.top_k}")
    print(f"Input: bs={args.bs}, slen={args.slen}")
    print(f"Simulated comm delay: {args.delay_ms}ms (dispatch + combine)")
    print()

    # Create ScMoE module
    moe_args = MoEArgs(
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts,
        top_k=args.top_k,
        use_scmoe=True,
        use_grouped_mm=True,
        score_func="softmax",
        score_before_experts=True,
        scmoe=ScMoEConfig(),
    )
    scmoe = ScMoE(moe_args=moe_args, dim=args.dim, hidden_dim=args.hidden).to(device, dtype)
    with torch.no_grad():
        scmoe.init_weights(0.02, device, 1)

    # Create inputs
    x_current = torch.randn(args.bs, args.slen, args.dim, device=device, dtype=dtype)
    x_shortcut = torch.randn(args.bs, args.slen, args.dim, device=device, dtype=dtype)

    # First measure shared_experts time alone
    x_flat = x_current.view(-1, args.dim)
    x_normed = scmoe.shared_norm(x_flat)
    for _ in range(5):
        scmoe._shared_forward(x_normed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        scmoe._shared_forward(x_normed)
    torch.cuda.synchronize()
    shared_ms = (time.perf_counter() - t0) / 20 * 1000

    # Measure routed experts time alone
    x_routed = scmoe.routed_norm(x_shortcut.view(-1, args.dim))
    for _ in range(5):
        scmoe._routed_forward(x_routed, args.bs, args.slen, args.dim)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        scmoe._routed_forward(x_routed, args.bs, args.slen, args.dim)
    torch.cuda.synchronize()
    routed_ms = (time.perf_counter() - t0) / 20 * 1000

    print(f"SharedExperts compute:  {shared_ms:.2f}ms")
    print(f"RoutedExperts compute:  {routed_ms:.2f}ms")
    print(f"Comm delay (each):     {args.delay_ms:.1f}ms")
    print()

    # Benchmark standard MoE (sequential)
    std_ms = benchmark_standard_moe(scmoe, x_current, delay_cycles, args.warmup, args.iters)

    # Benchmark ScMoE (overlap)
    sc_ms = benchmark_scmoe_overlap(scmoe, x_current, x_shortcut, delay_cycles, args.warmup, args.iters)

    print(f"Standard MoE (sequential): {std_ms:.2f}ms/iter")
    print(f"  = dispatch({args.delay_ms}ms) + routed({routed_ms:.1f}ms) + combine({args.delay_ms}ms) + shared({shared_ms:.1f}ms)")
    print(f"  expected ≈ {args.delay_ms + routed_ms + args.delay_ms + shared_ms:.1f}ms")
    print()
    print(f"ScMoE (overlap):           {sc_ms:.2f}ms/iter")
    print(f"  = max(dispatch, shared) + routed + combine")
    print(f"  expected ≈ {max(args.delay_ms, shared_ms) + routed_ms + args.delay_ms:.1f}ms")
    print()

    speedup = std_ms / sc_ms
    saved = std_ms - sc_ms
    print(f"Speedup: {speedup:.2f}x  ({saved:.1f}ms saved per layer)")
    print(f"Comm hidden by shared_experts: {min(shared_ms, args.delay_ms):.1f}ms of {args.delay_ms:.1f}ms dispatch")


if __name__ == "__main__":
    main()
