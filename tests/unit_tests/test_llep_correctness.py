#!/usr/bin/env python3
# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for LLEP optimization implementations.

Tests that the fused Triton kernel optimizations produce
numerically correct results compared to reference implementations.

Test categories:
1. Triton fused_silu_gate kernel (single-process, 4 tests)
2. Numerical stability (4 tests)
3. Performance benchmarks (1 test)

Run with torchrun (requires >= 1 GPU):
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py

Run specific test:
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py --test fused_silu_gate

List all tests:
    torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py --list
"""

import argparse
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Category 1: Triton Kernel Unit Tests (single-process)
# ---------------------------------------------------------------------------
def test_fused_silu_gate_vs_pytorch():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate vs PyTorch\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)
    x3 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)

    ref = F.silu(x1.float()) * x3.float()
    ref = ref.to(torch.bfloat16)

    out = fused_silu_gate(x1, x3)
    max_diff = (out - ref).abs().max().item()

    if rank == 0:
        print(f"  max_diff={max_diff:.6f}")
    assert max_diff < 1e-3, f"max_diff={max_diff:.6f}"
    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_backward():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate backward\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    x3 = torch.randn(32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    x1_ref = x1.detach().clone().requires_grad_(True)
    x3_ref = x3.detach().clone().requires_grad_(True)

    # Forward
    out_fused = fused_silu_gate(x1, x3)
    out_ref = (F.silu(x1_ref.float()) * x3_ref.float()).to(torch.bfloat16)

    # Backward
    grad_out = torch.randn_like(out_fused)
    out_fused.backward(grad_out)
    out_ref.backward(grad_out.clone())

    grad_x1_diff = (x1.grad - x1_ref.grad).abs().max().item()
    grad_x3_diff = (x3.grad - x3_ref.grad).abs().max().item()

    if rank == 0:
        print(f"  grad_x1 max_diff={grad_x1_diff:.6f}")
        print(f"  grad_x3 max_diff={grad_x3_diff:.6f}")

    assert grad_x1_diff < 2e-2, f"grad_x1 diff={grad_x1_diff:.6f}"
    assert grad_x3_diff < 2e-2, f"grad_x3 diff={grad_x3_diff:.6f}"

    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_various_shapes():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate various shapes\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    for num_tokens, hidden_size in [
        (32, 2048),
        (256, 2048),
        (1024, 2048),
        (4096, 2048),
    ]:
        torch.manual_seed(42)
        x1 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        x3 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        ref = (F.silu(x1.float()) * x3.float()).to(torch.bfloat16)
        out = fused_silu_gate(x1, x3)
        max_diff = (out - ref).abs().max().item()

        if rank == 0:
            print(f"  ({num_tokens}, {hidden_size}): max_diff={max_diff:.6f}")
        assert (
            max_diff < 1e-3
        ), f"shape=({num_tokens},{hidden_size}): max_diff={max_diff:.6f}"

    if rank == 0:
        print(f"  PASSED")


def test_fused_silu_gate_determinism():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Triton: fused_silu_gate determinism\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    torch.manual_seed(42)
    x1 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)
    x3 = torch.randn(256, 2048, device=device, dtype=torch.bfloat16)

    outputs = [fused_silu_gate(x1.clone(), x3.clone()) for _ in range(5)]
    all_match = all(torch.equal(outputs[0], outputs[i]) for i in range(1, 5))

    if rank == 0:
        print(f"  5 runs bitwise identical: {all_match}")
    assert all_match, "Determinism check failed"
    if rank == 0:
        print(f"  PASSED")


# ---------------------------------------------------------------------------
# Category 2: Numerical Stability
# ---------------------------------------------------------------------------
def test_numerical_bf16_vs_fp32():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: bf16 vs fp32 reference\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.float32) * 0.1
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    # fp32 reference
    out_fp32 = _llep_swiglu_ffn_forloop(
        x.clone(),
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    # bf16
    out_bf16 = _llep_swiglu_ffn_forloop(
        x.bfloat16(),
        expert_ids,
        w1.bfloat16(),
        w2.bfloat16(),
        w3.bfloat16(),
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    max_diff = (out_fp32 - out_bf16.float()).abs().max().item()
    if rank == 0:
        print(f"  bf16 vs fp32: max_diff={max_diff:.6f}")
    assert max_diff < 0.05, f"bf16 vs fp32 too large: {max_diff:.6f}"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_large_values():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: Large input values\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    x = (
        torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16) * 10.0
    )  # Large values
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out = _llep_swiglu_ffn_forloop(
        x,
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    if rank == 0:
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
    assert not has_nan, "Output contains NaN"
    assert not has_inf, "Output contains Inf"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_near_zero():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: Near-zero input values\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16) * 1e-6
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out = _llep_swiglu_ffn_forloop(
        x,
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    has_nan = torch.isnan(out).any().item()
    if rank == 0:
        print(f"  NaN: {has_nan}")
    assert not has_nan, "Output contains NaN for near-zero input"
    if rank == 0:
        print(f"  PASSED")


def test_numerical_fp32_precision():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Numerical: fp32 precision\n{'='*70}")

    from torchtitan.distributed.llep import _llep_swiglu_ffn_forloop

    torch.manual_seed(42)
    dim, hidden_dim, num_local_experts, num_tokens = 64, 128, 4, 32

    w1 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            num_local_experts, dim, hidden_dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    w3 = (
        torch.randn(
            num_local_experts, hidden_dim, dim, device=device, dtype=torch.float32
        )
        * 0.02
    )
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.float32) * 0.1
    expert_ids = torch.arange(num_tokens, device=device) % num_local_experts

    out_forloop = _llep_swiglu_ffn_forloop(
        x.clone(),
        expert_ids,
        w1,
        w2,
        w3,
        None,
        None,
        None,
        0,
        num_local_experts,
    )

    # Manually compute reference
    out_ref = torch.empty_like(x)
    for i in range(num_tokens):
        eid = expert_ids[i].item()
        xi = x[i : i + 1]
        h = F.silu(xi @ w1[eid].T) * (xi @ w3[eid].T)
        out_ref[i : i + 1] = h @ w2[eid].T

    max_diff = (out_forloop - out_ref).abs().max().item()
    if rank == 0:
        print(f"  fp32 forloop vs manual: max_diff={max_diff:.8f}")
    assert max_diff < 1e-5, f"fp32 precision: max_diff={max_diff:.8f}"
    if rank == 0:
        print(f"  PASSED")


# ---------------------------------------------------------------------------
# Category 3: Performance Benchmarks
# ---------------------------------------------------------------------------
def test_benchmark_triton_silu_gate():
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}\n  Benchmark: Triton fused_silu_gate vs F.silu*x3\n{'='*70}")

    from torchtitan.distributed.llep_kernels import fused_silu_gate

    configs = [(256, 2048), (1024, 2048), (4096, 2048)]

    for num_tokens, hidden_size in configs:
        torch.manual_seed(42)
        x1 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        x3 = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            F.silu(x1) * x3
            fused_silu_gate(x1, x3)
        torch.cuda.synchronize()

        # Benchmark unfused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_unfused = []
        for _ in range(20):
            start.record()
            _ = F.silu(x1) * x3
            end.record()
            torch.cuda.synchronize()
            times_unfused.append(start.elapsed_time(end))

        # Benchmark fused
        times_fused = []
        for _ in range(20):
            start.record()
            _ = fused_silu_gate(x1, x3)
            end.record()
            torch.cuda.synchronize()
            times_fused.append(start.elapsed_time(end))

        median_unfused = sorted(times_unfused)[10]
        median_fused = sorted(times_fused)[10]
        speedup = median_unfused / max(median_fused, 1e-6)

        if rank == 0:
            print(
                f"  ({num_tokens}, {hidden_size}): "
                f"unfused={median_unfused:.4f}ms  fused={median_fused:.4f}ms  "
                f"speedup={speedup:.2f}x"
            )

    if rank == 0:
        print(f"  DONE (benchmark only, no assertions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TEST_REGISTRY = {
    # Category 1: Triton kernel unit tests
    "fused_silu_gate": test_fused_silu_gate_vs_pytorch,
    "fused_silu_gate_backward": test_fused_silu_gate_backward,
    "fused_silu_gate_shapes": test_fused_silu_gate_various_shapes,
    "fused_silu_gate_determinism": test_fused_silu_gate_determinism,
    # Category 2: Numerical stability
    "numerical_bf16_vs_fp32": test_numerical_bf16_vs_fp32,
    "numerical_large_values": test_numerical_large_values,
    "numerical_near_zero": test_numerical_near_zero,
    "numerical_fp32_precision": test_numerical_fp32_precision,
    # Category 3: Benchmarks
    "benchmark_triton": test_benchmark_triton_silu_gate,
}


def main():
    parser = argparse.ArgumentParser(description="LLEP Correctness Tests")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run a specific test (e.g., 'fused_silu_gate'). If not specified, run all.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit."
    )
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name in TEST_REGISTRY:
            print(f"  {name}")
        sys.exit(0)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'#'*70}")
        print(f"  LLEP Correctness Test Suite")
        print(f"  world_size={world_size}")
        print(f"{'#'*70}")

    if args.test:
        if args.test not in TEST_REGISTRY:
            if rank == 0:
                print(f"Unknown test: {args.test}")
                print(f"Available: {list(TEST_REGISTRY.keys())}")
            dist.destroy_process_group()
            sys.exit(1)
        tests = {args.test: TEST_REGISTRY[args.test]}
    else:
        tests = TEST_REGISTRY

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests.items():
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            if rank == 0:
                print(f"\n  FAILED: {name}")
                traceback.print_exc()
            # Continue to next test
            dist.barrier()

    if rank == 0:
        print(f"\n{'#'*70}")
        print(
            f"  Results: {passed} passed, {failed} failed out of {passed + failed} tests"
        )
        if errors:
            print(f"\n  Failed tests:")
            for name, err in errors:
                print(f"    {name}: {err}")
        print(f"{'#'*70}\n")

    dist.destroy_process_group()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
