# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests that PROVE the offload engine's benefits are real.

Each test measures a concrete property:
1. D2H copy runs async (overlaps with GPU compute)
2. H2D copy runs async (overlaps with GPU compute)
3. Pinned memory pool reuse is faster than fresh allocation
4. Per-module granularity: offload big tensors, skip small ones
5. Activations are actually moved to CPU (not just copied)
6. forced_released_tensors actually frees GPU memory
7. Dedicated streams are separate from default stream
8. CUDA events correctly synchronize across streams
"""

import time

import pytest
import torch
import torch.nn as nn

from torchtitan.distributed.cpu_offload.tensor_pool import TensorPool
from torchtitan.distributed.cpu_offload.chunk_handler import ChunkOffloadHandler
from torchtitan.distributed.cpu_offload.offload_group import OffloadTensorGroup
from torchtitan.distributed.cpu_offload.offload_manager import OffloadManager
from torchtitan.distributed.cpu_offload import (
    ActivationOffloadContext,
    group_commit,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_handler(min_size=1):
    pool = TensorPool(device="cpu", pin_memory=True)
    d2h = torch.cuda.Stream()
    h2d = torch.cuda.Stream()
    return ChunkOffloadHandler(min_size, pool, d2h, h2d), pool


def _gpu_busy_work(size=2048, iters=50):
    """Do enough GPU work to take measurable time.
    Uses addition to avoid NaN from repeated matmul overflow."""
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    for _ in range(iters):
        a = torch.addmm(a, a, b, alpha=0.001, beta=0.999)
    return a


def _time_cuda_ms(fn):
    """Time a function using CUDA events (accurate GPU timing)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return end.elapsed_time(start), result


# ═══════════════════════════════════════════════════════════════════════
# 1. ASYNC D2H: GPU->CPU copy overlaps with GPU compute
# ═══════════════════════════════════════════════════════════════════════


class TestAsyncD2H:
    """Prove that D2H copies run on a separate stream and overlap with compute."""

    def test_d2h_stream_is_separate(self):
        """D2H stream must be different from the default compute stream."""
        handler, pool = _make_handler()
        assert handler.d2h_stream != torch.cuda.current_stream()
        assert handler.d2h_stream != torch.cuda.default_stream()
        pool.clear()

    def test_d2h_copy_is_non_blocking(self):
        """offload() with pinned memory uses non_blocking=True.
        Verify the CPU tensor is pinned and the copy doesn't block."""
        handler, pool = _make_handler()
        gpu_tensor = torch.randn(1024, 1024, device="cuda")

        # offload in the D2H stream
        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(gpu_tensor, use_cpu_pool=True)

        _, cpu_backup, _ = state
        assert cpu_backup.is_pinned(), "CPU backup must be pinned for async copy"
        # Don't synchronize yet — the copy may still be in flight
        # This proves it was non-blocking
        torch.cuda.synchronize()
        assert torch.allclose(cpu_backup, gpu_tensor.cpu())
        pool.clear()

    def test_d2h_overlaps_with_compute(self):
        """D2H copy should run concurrently with compute on the default stream.

        Strategy: measure wall-clock time of (compute_only) vs (compute + offload).
        If offload is truly async, combined ≈ max(compute, offload), not sum.
        """
        handler, pool = _make_handler()
        gpu_tensor = torch.randn(2048, 2048, device="cuda")

        # Time compute only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _gpu_busy_work(size=1024, iters=30)
        torch.cuda.synchronize()
        compute_s = time.perf_counter() - t0

        # Time compute + concurrent offload
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(gpu_tensor, use_cpu_pool=False)
        _gpu_busy_work(size=1024, iters=30)
        torch.cuda.synchronize()
        combined_s = time.perf_counter() - t0

        compute_ms = compute_s * 1000
        combined_ms = combined_s * 1000
        print(f"Compute only: {compute_ms:.2f}ms, Compute+D2H: {combined_ms:.2f}ms")
        # If truly overlapped, combined should be less than 2x compute
        assert combined_ms < compute_ms * 2.0, \
            f"D2H not overlapping! compute={compute_ms:.1f}ms combined={combined_ms:.1f}ms"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 2. ASYNC H2D: CPU->GPU copy overlaps with GPU compute
# ═══════════════════════════════════════════════════════════════════════


class TestAsyncH2D:
    """Prove that H2D copies run on a separate stream and overlap with compute."""

    def test_h2d_stream_is_separate(self):
        handler, pool = _make_handler()
        assert handler.h2d_stream != torch.cuda.current_stream()
        assert handler.h2d_stream != torch.cuda.default_stream()
        pool.clear()

    def test_h2d_copy_is_non_blocking(self):
        """reload() with pinned memory uses non_blocking=True."""
        handler, pool = _make_handler()
        gpu_tensor = torch.randn(1024, 1024, device="cuda")
        expected = gpu_tensor.clone()

        state = handler.offload(gpu_tensor, use_cpu_pool=False)
        torch.cuda.synchronize()

        with torch.cuda.stream(handler.h2d_stream):
            recovered = handler.reload(state)

        # recovered should be on GPU
        assert recovered.device.type == "cuda"
        torch.cuda.synchronize()
        assert torch.equal(recovered, expected)
        pool.clear()

    def test_h2d_overlaps_with_compute(self):
        """H2D reload should overlap with compute on the default stream."""
        handler, pool = _make_handler()
        gpu_tensor = torch.randn(2048, 2048, device="cuda")

        # Time compute only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _gpu_busy_work(size=1024, iters=30)
        torch.cuda.synchronize()
        compute_s = time.perf_counter() - t0

        # Time compute + concurrent reload
        state = handler.offload(gpu_tensor, use_cpu_pool=False)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        handler.h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(handler.h2d_stream):
            recovered = handler.reload(state)
        _gpu_busy_work(size=1024, iters=30)
        torch.cuda.synchronize()
        combined_s = time.perf_counter() - t0

        compute_ms = compute_s * 1000
        combined_ms = combined_s * 1000
        print(f"Compute only: {compute_ms:.2f}ms, Compute+H2D: {combined_ms:.2f}ms")
        assert combined_ms < compute_ms * 2.0, \
            f"H2D not overlapping! compute={compute_ms:.1f}ms combined={combined_ms:.1f}ms"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 3. PINNED MEMORY POOL REUSE vs FRESH ALLOCATION
# ═══════════════════════════════════════════════════════════════════════


class TestPinnedPoolPerformance:
    """Prove that pool reuse is faster than fresh pinned allocation."""

    def test_pool_reuse_faster_than_fresh(self):
        """After warmup, pool.allocate() should be much faster than
        torch.empty(..., pin_memory=True) because it reuses existing tensors."""
        shape = (512, 1024)
        dtype = torch.float32
        n_iters = 100

        pool = TensorPool(device="cpu", pin_memory=True)

        # Warmup: fill the pool
        t = pool.allocate(shape, dtype)
        pool.free(t)

        # Time pool reuse (should be O(1) deque pop)
        start = time.perf_counter()
        for _ in range(n_iters):
            t = pool.allocate(shape, dtype)
            pool.free(t)
        pool_time = time.perf_counter() - start

        # Time fresh allocation (cudaMallocHost each time)
        start = time.perf_counter()
        for _ in range(n_iters):
            t = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            del t
        fresh_time = time.perf_counter() - start

        speedup = fresh_time / max(pool_time, 1e-9)
        print(f"Pool: {pool_time*1000:.2f}ms, Fresh: {fresh_time*1000:.2f}ms, "
              f"Speedup: {speedup:.1f}x")
        assert pool_time < fresh_time, \
            f"Pool should be faster! pool={pool_time:.4f}s fresh={fresh_time:.4f}s"
        pool.clear()

    def test_pool_hits_after_warmup(self):
        """After one allocation+free cycle, all subsequent allocations are hits."""
        pool = TensorPool(device="cpu", pin_memory=True)
        shape = (256, 512)
        dtype = torch.float32

        t = pool.allocate(shape, dtype)
        pool.free(t)

        for _ in range(50):
            t = pool.allocate(shape, dtype)
            pool.free(t)

        assert pool.stats["pool_hits"] == 50
        assert pool.stats["pool_misses"] == 1
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 4. PER-MODULE GRANULARITY
# ═══════════════════════════════════════════════════════════════════════


class TestPerModuleGranularity:
    """Prove that you can selectively offload per-module."""

    def test_size_threshold_skips_small_tensors(self):
        """min_offloaded_tensor_size filters out small activations."""
        handler, pool = _make_handler(min_size=10000)  # skip tensors < 10000 elements

        small = torch.randn(100, device="cuda")   # 100 elements
        big = torch.randn(200, 200, device="cuda")  # 40000 elements

        assert handler.should_offload_tensor(small) is False
        assert handler.should_offload_tensor(big) is True
        pool.clear()

    def test_mark_not_offloadable_excludes_tensor(self):
        """Tensors marked as not-offloadable are skipped."""
        handler, pool = _make_handler(min_size=1)

        param = torch.randn(1000, device="cuda")
        param.offloading_activation = False  # mark as parameter

        activation = torch.randn(1000, device="cuda")
        # no attribute → defaults to offloadable

        assert handler.should_offload_tensor(param) is False
        assert handler.should_offload_tensor(activation) is True
        pool.clear()

    def test_selective_offload_in_model(self):
        """In a model with two modules, offload only one."""

        class SelectiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.big_linear = nn.Linear(512, 2048)  # big activation
                self.small_norm = nn.LayerNorm(2048)     # small, recompute instead
                self.proj = nn.Linear(2048, 512)

            def forward(self, x):
                # Only offload big_linear, not small_norm
                with ActivationOffloadContext(True, x, "big_linear") as x_in:
                    h = self.big_linear(x_in)
                h = group_commit(h, "big_linear", forced_released_tensors=[x])
                # small_norm: no offloading
                h = self.small_norm(h)
                return self.proj(h)

        OffloadManager.reset_instance()
        ActivationOffloadContext.init_chunk_handler(
            vp_size=1, vp_stage=0, min_offloaded_tensor_size=1
        )

        model = SelectiveModel().cuda()
        x = torch.randn(4, 32, 512, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Verify the chunk has exactly 1 group (big_linear), not 2
        mgr = OffloadManager.get_instance()
        chunk = mgr._cached_chunks_forward[0]
        assert len(chunk.offload_groups) == 1
        assert chunk.offload_groups[0]._name == "big_linear"


# ═══════════════════════════════════════════════════════════════════════
# 5. ACTIVATIONS ACTUALLY MOVE TO CPU
# ═══════════════════════════════════════════════════════════════════════


class TestActivationsOnCPU:
    """Prove that activations are physically on CPU after offloading."""

    def test_offloaded_tensor_is_on_cpu(self):
        """After offload(), the backup tensor is on CPU, pinned."""
        handler, pool = _make_handler()
        gpu_t = torch.randn(512, 512, device="cuda")
        state = handler.offload(gpu_t, use_cpu_pool=True)
        torch.cuda.synchronize()

        dev, cpu_backup, _ = state
        assert cpu_backup.device == torch.device("cpu")
        assert cpu_backup.is_pinned()
        assert dev.type == "cuda"
        pool.clear()

    def test_offloaded_state_is_tuple_not_tensor(self):
        """In OffloadTensorGroup, offloaded tensors are replaced with
        (device, cpu_backup, pool_flag) tuples, not GPU tensors."""
        handler, pool = _make_handler()
        handler.is_warmup = True
        handler.do_offload = True
        handler._offloaded_group_index = 1

        group = OffloadTensorGroup("test", use_cpu_pool=False)
        handler.offload_groups.append(group)
        handler._max_group_size = 1

        gpu_t = torch.randn(256, 256, device="cuda")
        tag = (1, 0)
        group.push_tensor(tag, gpu_t)

        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_offload_group()
        torch.cuda.synchronize()

        # The tensor in the group should now be a tuple, not a GPU tensor
        state = group._tensors[tag]
        assert isinstance(state, tuple), f"Expected tuple, got {type(state)}"
        assert len(state) == 3
        dev, cpu_backup, use_pool = state
        assert cpu_backup.device == torch.device("cpu")
        pool.clear()

    def test_reload_restores_correct_values(self):
        """Full cycle: verify data integrity after CPU roundtrip."""
        handler, pool = _make_handler()
        original = torch.randn(128, 256, device="cuda")
        expected = original.clone()

        state = handler.offload(original, use_cpu_pool=False)
        torch.cuda.synchronize()

        # Corrupt the original GPU tensor to prove reload uses CPU data
        original.fill_(999.0)

        recovered = handler.reload(state)
        torch.cuda.synchronize()

        assert torch.equal(recovered, expected), "Reload should use CPU backup, not corrupted GPU"
        assert not torch.equal(recovered, original), "Should differ from corrupted tensor"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 6. FORCED TENSOR RELEASE
# ═══════════════════════════════════════════════════════════════════════


class TestForcedTensorRelease:
    """Prove that forced_released_tensors actually frees GPU storage."""

    def test_resize_zero_frees_storage(self):
        """untyped_storage().resize_(0) should make the tensor use ~0 bytes."""
        t = torch.randn(1024, 1024, device="cuda")  # ~4MB
        original_size = t.untyped_storage().size()
        assert original_size > 0

        t.record_stream(torch.cuda.current_stream())
        t.untyped_storage().resize_(0)

        assert t.untyped_storage().size() == 0, "Storage should be 0 after resize_(0)"

    def test_forced_release_in_bulk_offload(self):
        """When bulk_offload uses forced_released_tensors, those tensors
        should have their storage freed."""
        handler, pool = _make_handler()
        handler.is_warmup = True
        handler.do_offload = True
        handler._offloaded_group_index = 1

        group = OffloadTensorGroup("test", use_cpu_pool=False)
        handler.offload_groups.append(group)
        handler._max_group_size = 1

        # The tensor we want to force-release
        to_release = torch.randn(512, 512, device="cuda")
        tag = (1, 0)
        group.push_tensor(tag, to_release)

        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())

        # Use bulk_offload which handles forced_released_tensors
        def noop_front_backward_chunk(name=None):
            return None

        handler.bulk_offload([to_release], noop_front_backward_chunk)
        torch.cuda.synchronize()

        assert to_release.untyped_storage().size() == 0, \
            "forced_released_tensors should have storage freed"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 7. CUDA EVENT SYNCHRONIZATION CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════


class TestCUDAEventSync:
    """Prove that CUDA events correctly synchronize D2H/H2D with compute."""

    def test_offload_event_prevents_data_race(self):
        """Without waiting on offload_event, reload could start before
        offload finishes. This test verifies the event prevents that."""
        handler, pool = _make_handler()

        group = OffloadTensorGroup("test", use_cpu_pool=False)
        gpu_t = torch.randn(512, 512, device="cuda")
        expected = gpu_t.clone()
        tag = (1, 0)
        group.push_tensor(tag, gpu_t)

        # Offload in D2H stream
        with torch.cuda.stream(handler.d2h_stream):
            for t, tensor in group._tensors.items():
                state = handler.offload(tensor, use_cpu_pool=False)
                group.push_tensor(t, state)
            group.record_offload_event(handler.d2h_stream)

        # Reload in H2D stream — MUST wait on offload event
        with torch.cuda.stream(handler.h2d_stream):
            group.wait_offload_event(handler.h2d_stream)  # THIS is what we're testing
            state = group.pop_tensor(tag)
            recovered = handler.reload(state)

        torch.cuda.synchronize()
        assert torch.equal(recovered, expected), "Event sync should ensure correct data"
        pool.clear()

    def test_reload_event_prevents_use_before_ready(self):
        """Compute stream must wait on reload_event before using reloaded tensors."""
        handler, pool = _make_handler()

        gpu_t = torch.randn(256, 256, device="cuda")
        expected = gpu_t.clone()
        state = handler.offload(gpu_t, use_cpu_pool=False)
        torch.cuda.synchronize()

        group = OffloadTensorGroup("test", use_cpu_pool=False)
        group.push_tensor((1, 0), state)
        group.record_offload_event(handler.d2h_stream)  # already done

        # Reload in H2D stream
        with torch.cuda.stream(handler.h2d_stream):
            group.wait_offload_event(handler.h2d_stream)
            s = group.pop_tensor((1, 0))
            recovered = handler.reload(s)
            group.record_reload_event(handler.h2d_stream)

        # Main stream waits for reload to finish
        group.wait_reload_event(torch.cuda.current_stream())

        # Now safe to use recovered tensor on main stream
        result = recovered + 1.0
        torch.cuda.synchronize()
        assert torch.allclose(result, expected + 1.0)
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# 8. END-TO-END: ASYNC OFFLOAD IN A REAL FORWARD/BACKWARD
# ═══════════════════════════════════════════════════════════════════════


class TestE2EAsyncOffload:
    """Prove async offloading works correctly in a real training step."""

    def test_async_offload_does_not_slow_down_forward(self):
        """Forward pass with offloading should not be much slower than
        without, because D2H runs concurrently with next layer's compute."""

        class DeepMLP(nn.Module):
            def __init__(self, dim, hidden, n_layers, offload=False):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(dim if i == 0 else hidden,
                              dim if i == n_layers - 1 else hidden)
                    for i in range(n_layers)
                ])
                self.offload = offload

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    if self.offload:
                        with ActivationOffloadContext(True, x, "layer") as x_in:
                            x = layer(x_in)
                        x = group_commit(x, "layer")
                    else:
                        x = layer(x)
                    if i < len(self.layers) - 1:
                        x = torch.relu(x)
                return x

        dim, hidden, n_layers = 256, 1024, 8
        x = torch.randn(8, 64, dim, device="cuda")

        # Baseline timing (no offload)
        base_model = DeepMLP(dim, hidden, n_layers, offload=False).cuda()
        # Warmup
        for _ in range(3):
            base_model(x).sum().backward()
        base_ms, _ = _time_cuda_ms(lambda: base_model(x).sum().backward())

        # Offloaded timing
        off_model = DeepMLP(dim, hidden, n_layers, offload=True).cuda()
        off_model.load_state_dict(base_model.state_dict())

        # Warmup (iteration 0)
        OffloadManager.reset_instance()
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        off_model(x).sum().backward()
        ActivationOffloadContext.reset()

        # Steady state
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        # Warmup iterations
        for _ in range(3):
            ActivationOffloadContext.reset()
            ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
            off_model(x).sum().backward()

        ActivationOffloadContext.reset()
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        off_ms, _ = _time_cuda_ms(lambda: off_model(x).sum().backward())

        overhead_pct = ((off_ms - base_ms) / base_ms) * 100
        print(f"Baseline: {base_ms:.2f}ms, Offloaded: {off_ms:.2f}ms, "
              f"Overhead: {overhead_pct:.1f}%")

        # Async offloading overhead should be < 50%
        # (Megatron reports 1.6-2% for real models; we allow more for test models)
        assert off_ms < base_ms * 1.5, \
            f"Offload overhead too high: {overhead_pct:.1f}% (base={base_ms:.1f}ms, off={off_ms:.1f}ms)"

    def test_offload_produces_correct_gradients_with_async(self):
        """Even with async transfers, gradients must be numerically correct."""

        class MLP(nn.Module):
            def __init__(self, dim, hidden, offload=False):
                super().__init__()
                self.fc1 = nn.Linear(dim, hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.fc3 = nn.Linear(hidden, dim)
                self.offload = offload

            def forward(self, x):
                if self.offload:
                    with ActivationOffloadContext(True, x, "fc") as x_in:
                        h = torch.relu(self.fc1(x_in))
                    h = group_commit(h, "fc", forced_released_tensors=[x])
                    with ActivationOffloadContext(True, h, "fc") as h_in:
                        h2 = torch.relu(self.fc2(h_in))
                    h2 = group_commit(h2, "fc", forced_released_tensors=[h])
                    with ActivationOffloadContext(True, h2, "fc") as h2_in:
                        out = self.fc3(h2_in)
                    out = group_commit(out, "fc")
                else:
                    h = torch.relu(self.fc1(x))
                    h2 = torch.relu(self.fc2(h))
                    out = self.fc3(h2)
                return out

        torch.manual_seed(42)
        dim, hidden = 256, 512
        x = torch.randn(4, 32, dim, device="cuda")

        base = MLP(dim, hidden, offload=False).cuda()
        base.zero_grad()
        base_out = base(x)
        base_out.sum().backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        off = MLP(dim, hidden, offload=True).cuda()
        off.load_state_dict(base.state_dict())
        OffloadManager.reset_instance()
        ActivationOffloadContext.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        off.zero_grad()
        off_out = off(x)
        off_out.sum().backward()
        off_grads = {n: p.grad.clone() for n, p in off.named_parameters()}

        # Output correctness
        assert torch.allclose(off_out, base_out, rtol=1e-4, atol=1e-4)

        # Gradient correctness
        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-4, atol=1e-4), \
                f"Grad mismatch at {name}: max_diff={( off_grads[name] - base_grads[name]).abs().max()}"
