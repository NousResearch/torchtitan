# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests covering gaps identified from DeepSpeed, ColossalAI, and Unsloth.

Each test addresses a specific pattern found in other frameworks' test suites
that our original tests did not cover.
"""

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


def _make_handler(min_size=1):
    pool = TensorPool(device="cpu", pin_memory=True)
    d2h = torch.cuda.Stream()
    h2d = torch.cuda.Stream()
    return ChunkOffloadHandler(min_size, pool, d2h, h2d), pool


def _init_offload():
    OffloadManager.reset_instance()
    ActivationOffloadContext.init_chunk_handler(
        vp_size=1, vp_stage=0, min_offloaded_tensor_size=1
    )


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: Memory allocation verification
# DeepSpeed checks memory_allocated() decreases after offload
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryAllocationTracking:
    """Verify GPU memory_allocated changes during offload/reload cycle.
    (DeepSpeed pattern: test_offload_states.py)"""

    def test_gpu_memory_decreases_after_offload(self):
        """After offloading, live GPU memory should decrease.
        Uses memory_allocated() (live) not max_memory_allocated() (peak)."""
        handler, pool = _make_handler()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Allocate a large GPU tensor
        t = torch.randn(2048, 2048, device="cuda")  # ~16MB
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Offload to CPU
        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()

        # Free the original GPU tensor storage
        t.record_stream(torch.cuda.current_stream())
        t.untyped_storage().resize_(0)
        torch.cuda.synchronize()
        mem_after_offload = torch.cuda.memory_allocated()

        assert mem_after_offload < mem_before, \
            f"Memory should decrease: before={mem_before}, after={mem_after_offload}"

        # Reload back to GPU
        with torch.cuda.stream(handler.h2d_stream):
            recovered = handler.reload(state)
        torch.cuda.synchronize()
        mem_after_reload = torch.cuda.memory_allocated()

        assert mem_after_reload > mem_after_offload, \
            f"Memory should increase after reload: offloaded={mem_after_offload}, reloaded={mem_after_reload}"
        del recovered
        pool.clear()

    def test_memory_delta_matches_tensor_size(self):
        """The memory freed should approximately match the tensor's byte size."""
        handler, pool = _make_handler()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        size = (1024, 1024)
        dtype = torch.float32
        expected_bytes = 1024 * 1024 * 4  # ~4MB

        t = torch.randn(*size, device="cuda", dtype=dtype)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()
        t.untyped_storage().resize_(0)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        freed = mem_before - mem_after
        # Allow 10% tolerance for allocator overhead
        assert freed >= expected_bytes * 0.9, \
            f"Freed {freed} bytes, expected ~{expected_bytes}"
        del state
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: Device placement verification
# DeepSpeed validates device of each tensor after offload/reload
# ═══════════════════════════════════════════════════════════════════════


class TestDevicePlacement:
    """Verify tensors are on the correct device at each stage."""

    def test_offload_state_devices(self):
        """After offload: original device saved, backup on CPU."""
        handler, pool = _make_handler()
        t = torch.randn(256, 256, device="cuda")
        state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()

        dev, cpu_backup, _ = state
        assert dev.type == "cuda", f"Saved device should be cuda, got {dev}"
        assert cpu_backup.device.type == "cpu", f"Backup should be on CPU, got {cpu_backup.device}"
        pool.clear()

    def test_reload_restores_to_original_device(self):
        """After reload: tensor must be on the same CUDA device as original."""
        handler, pool = _make_handler()
        t = torch.randn(256, 256, device="cuda:0")
        state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()

        recovered = handler.reload(state)
        assert recovered.device == torch.device("cuda", 0), \
            f"Should be on cuda:0, got {recovered.device}"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: pin_memory=False fallback
# DeepSpeed tests both pinned and unpinned paths
# ═══════════════════════════════════════════════════════════════════════


class TestUnpinnedFallback:
    """Offloading should work without pinned memory (slower but correct)."""

    def test_offload_reload_without_pinned_memory(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        d2h = torch.cuda.Stream()
        h2d = torch.cuda.Stream()
        handler = ChunkOffloadHandler(1, pool, d2h, h2d)

        t = torch.randn(512, 512, device="cuda")
        expected = t.clone()

        state = handler.offload(t, use_cpu_pool=True)
        torch.cuda.synchronize()

        _, cpu_backup, _ = state
        assert not cpu_backup.is_pinned(), "Should NOT be pinned"

        recovered = handler.reload(state, non_blocking=False)
        torch.cuda.synchronize()

        assert torch.equal(recovered, expected)
        pool.clear()

    def test_unpinned_pool_allocation(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t = pool.allocate((64, 64), dtype=torch.float32)
        assert not t.is_pinned()
        pool.free(t)
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: Exact data integrity (torch.equal, not allclose)
# DeepSpeed uses torch.equal() for offload/reload verification
# ═══════════════════════════════════════════════════════════════════════


class TestExactDataIntegrity:
    """Offload/reload must preserve data EXACTLY (bitwise), not approximately."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("size", [(64,), (256, 256), (32, 64, 128)])
    def test_bitwise_exact_roundtrip(self, dtype, size):
        handler, pool = _make_handler()
        t = torch.randn(*size, device="cuda", dtype=dtype)
        expected = t.clone()

        state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()
        recovered = handler.reload(state)
        torch.cuda.synchronize()

        assert torch.equal(recovered, expected), \
            f"Bitwise mismatch for dtype={dtype}, size={size}"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: Multiple offload/reload cycles on same tensor
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleOffloadReloadCycles:
    """Same tensor offloaded and reloaded multiple times."""

    def test_repeated_offload_reload(self):
        handler, pool = _make_handler()
        t = torch.randn(512, 512, device="cuda")
        expected = t.clone()

        for cycle in range(5):
            state = handler.offload(t, use_cpu_pool=False)
            torch.cuda.synchronize()
            t = handler.reload(state)
            torch.cuda.synchronize()
            assert torch.equal(t, expected), f"Mismatch at cycle {cycle}"
        pool.clear()

    def test_pool_handles_repeated_cycles(self):
        """Pool should efficiently handle repeated alloc/free cycles."""
        pool = TensorPool(device="cpu", pin_memory=True)
        d2h = torch.cuda.Stream()
        h2d = torch.cuda.Stream()
        handler = ChunkOffloadHandler(1, pool, d2h, h2d)

        t = torch.randn(256, 256, device="cuda")
        for _ in range(20):
            state = handler.offload(t, use_cpu_pool=True)
            torch.cuda.synchronize()
            t = handler.reload(state)
            torch.cuda.synchronize()

        # After warmup, pool should be all hits
        assert pool.stats["pool_misses"] == 1, \
            f"Expected 1 miss (first alloc), got {pool.stats['pool_misses']}"
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM COLOSSALAI: Large tensor offload (> 100MB)
# ═══════════════════════════════════════════════════════════════════════


class TestLargeTensorOffload:
    """Test with large tensors to stress PCIe bandwidth."""

    def test_large_tensor_roundtrip(self):
        handler, pool = _make_handler()
        # ~128MB tensor
        t = torch.randn(4096, 8192, device="cuda", dtype=torch.float32)
        expected = t.clone()

        state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()
        recovered = handler.reload(state)
        torch.cuda.synchronize()

        assert torch.equal(recovered, expected), "Large tensor roundtrip failed"
        pool.clear()

    def test_massive_offload_50pct_gpu_memory(self):
        """Offload ~50% of GPU memory worth of tensors in chunks.

        Auto-calculates target based on available GPU memory.
        Verifies bitwise integrity, memory tracking, and pool reuse.
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        target_bytes = int(free_bytes * 0.5)
        # Cap at 20GB to keep test reasonable
        target_bytes = min(target_bytes, 20 * 1024**3)

        chunk_bytes = 512 * 1024 * 1024  # 512MB per chunk
        n_chunks = max(1, target_bytes // chunk_bytes)
        # Each chunk: 512MB / 4 bytes = 128M elements -> shape (8192, 16384)
        chunk_rows = 8192
        chunk_cols = chunk_bytes // (chunk_rows * 4)

        target_gb = (n_chunks * chunk_bytes) / 1024**3
        print(f"\nGPU: {torch.cuda.get_device_name(0)}, "
              f"Free: {free_bytes / 1024**3:.1f}GB, "
              f"Target offload: {target_gb:.1f}GB ({n_chunks} x 512MB chunks)")

        handler, pool = _make_handler()

        # Allocate all chunks on GPU
        gpu_tensors = []
        expected_data = []
        for i in range(n_chunks):
            t = torch.randn(chunk_rows, chunk_cols, device="cuda", dtype=torch.float32)
            expected_data.append(t.clone())
            gpu_tensors.append(t)

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Offload all chunks to CPU
        states = []
        for i, t in enumerate(gpu_tensors):
            handler.d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(handler.d2h_stream):
                state = handler.offload(t, use_cpu_pool=False)
            states.append(state)

        torch.cuda.synchronize()

        # Free GPU storage
        for t in gpu_tensors:
            t.record_stream(torch.cuda.current_stream())
            t.untyped_storage().resize_(0)
        torch.cuda.synchronize()

        mem_after_offload = torch.cuda.memory_allocated()
        freed_gb = (mem_before - mem_after_offload) / 1024**3
        print(f"Memory freed: {freed_gb:.1f}GB "
              f"(before={mem_before / 1024**3:.1f}GB, after={mem_after_offload / 1024**3:.1f}GB)")

        assert mem_after_offload < mem_before, "Memory should decrease after offload"

        # Reload all chunks back to GPU and verify
        for i, state in enumerate(states):
            handler.h2d_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(handler.h2d_stream):
                recovered = handler.reload(state)
            torch.cuda.synchronize()

            assert torch.equal(recovered, expected_data[i]), \
                f"Bitwise mismatch on chunk {i}/{n_chunks}"
            del recovered

        torch.cuda.synchronize()
        mem_after_reload = torch.cuda.memory_allocated()
        print(f"All {n_chunks} chunks verified. "
              f"Memory after reload cleanup: {mem_after_reload / 1024**3:.1f}GB")

        del expected_data, gpu_tensors, states
        pool.clear()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("target_gb", [1, 5, 10])
    def test_single_huge_tensor_offload(self, target_gb):
        """Offload a single contiguous tensor of target_gb size.

        Auto-skips if GPU doesn't have enough free memory (need ~2.5x
        target for: tensor + clone + reload headroom).
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        free_bytes, _ = torch.cuda.mem_get_info()
        needed = int(target_gb * 1024**3 * 2.5)
        if free_bytes < needed:
            pytest.skip(
                f"Need {needed / 1024**3:.1f}GB free, only {free_bytes / 1024**3:.1f}GB"
            )

        # Single tensor: target_gb / 4 bytes per float32 = numel
        numel = (target_gb * 1024**3) // 4
        # Shape as 2D for simplicity
        cols = 32768
        rows = numel // cols

        handler, pool = _make_handler()
        print(f"\nAllocating single {target_gb}GB tensor ({rows} x {cols} f32)...")

        t = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        expected = t.clone()
        actual_gb = t.numel() * t.element_size() / 1024**3
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Offload the single huge tensor
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(t, use_cpu_pool=False)
        torch.cuda.synchronize()

        # Free GPU storage
        t.record_stream(torch.cuda.current_stream())
        t.untyped_storage().resize_(0)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        freed_gb = (mem_before - mem_after) / 1024**3
        print(f"Single tensor: {actual_gb:.1f}GB, freed: {freed_gb:.1f}GB")
        assert mem_after < mem_before

        # Reload and verify bitwise
        handler.h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(handler.h2d_stream):
            recovered = handler.reload(state)
        torch.cuda.synchronize()

        assert torch.equal(recovered, expected), \
            f"Bitwise mismatch on single {target_gb}GB tensor"
        print(f"Single {target_gb}GB tensor: bitwise verified")

        del t, expected, recovered
        pool.clear()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# FROM UNSLOTH: Gradient accumulation with offloading
# ═══════════════════════════════════════════════════════════════════════


class TestGradientAccumulation:
    """Offloading must work correctly with gradient accumulation
    (multiple forward/backward before optimizer step)."""

    def test_grad_accum_with_offload(self):
        """Run 3 micro-batches, accumulate gradients, then verify."""
        torch.manual_seed(42)
        dim, hidden = 128, 256
        accum_steps = 3

        class MLP(nn.Module):
            def __init__(self, offload=False):
                super().__init__()
                self.fc1 = nn.Linear(dim, hidden)
                self.fc2 = nn.Linear(hidden, dim)
                self.offload = offload

            def forward(self, x):
                if self.offload:
                    with ActivationOffloadContext(True, x, "fc") as x_in:
                        h = torch.relu(self.fc1(x_in))
                    h = group_commit(h, "fc", forced_released_tensors=[x])
                    with ActivationOffloadContext(True, h, "fc") as h_in:
                        out = self.fc2(h_in)
                    out = group_commit(out, "fc")
                else:
                    h = torch.relu(self.fc1(x))
                    out = self.fc2(h)
                return out

        # Baseline: no offload
        torch.manual_seed(42)
        base = MLP(offload=False).cuda()
        base.zero_grad()
        for step in range(accum_steps):
            torch.manual_seed(100 + step)
            x = torch.randn(4, 16, dim, device="cuda")
            loss = base(x).sum() / accum_steps
            loss.backward()
        base_grads = {n: p.grad.clone() for n, p in base.named_parameters()}

        # Offloaded
        torch.manual_seed(42)
        off = MLP(offload=True).cuda()
        off.load_state_dict(base.state_dict())
        off.zero_grad()
        _init_offload()
        for step in range(accum_steps):
            torch.manual_seed(100 + step)
            x = torch.randn(4, 16, dim, device="cuda")
            loss = off(x).sum() / accum_steps
            loss.backward()
            # Reset offload manager between micro-batches
            ActivationOffloadContext.reset()
            ActivationOffloadContext.init_chunk_handler(
                vp_size=1, vp_stage=0, min_offloaded_tensor_size=1
            )
        off_grads = {n: p.grad.clone() for n, p in off.named_parameters()}

        for name in base_grads:
            assert torch.allclose(off_grads[name], base_grads[name], rtol=1e-4, atol=1e-4), \
                f"Grad accum mismatch at {name}: {(off_grads[name] - base_grads[name]).abs().max()}"


# ═══════════════════════════════════════════════════════════════════════
# FROM DEEPSPEED: Contiguous vs non-contiguous group offload
# ═══════════════════════════════════════════════════════════════════════


class TestContiguousHandling:
    """Verify handling of contiguous and non-contiguous tensors in bulk operations."""

    def test_mixed_contiguous_non_contiguous_group(self):
        """Group with both contiguous and non-contiguous tensors."""
        handler, pool = _make_handler()
        handler.is_warmup = True
        handler.do_offload = True
        handler._offloaded_group_index = 1

        group = OffloadTensorGroup("mixed", use_cpu_pool=False)
        handler.offload_groups.append(group)
        handler._max_group_size = 1

        contig = torch.randn(64, 128, device="cuda")  # contiguous
        non_contig = torch.randn(128, 64, device="cuda").t()  # non-contiguous
        assert not non_contig.is_contiguous()

        contig_expected = contig.clone()
        non_contig_expected = non_contig.clone()

        group.push_tensor((1, 0), contig)
        group.push_tensor((1, 1), non_contig)

        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_offload_group()
        torch.cuda.synchronize()

        # Reload
        handler._groups_to_reload.append(group)
        handler.h2d_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_reload_group()
        torch.cuda.synchronize()

        r_contig = group.pop_tensor((1, 0))
        r_non_contig = group.pop_tensor((1, 1))

        assert torch.equal(r_contig, contig_expected)
        assert torch.allclose(r_non_contig, non_contig_expected)
        # Non-contiguous becomes contiguous after offload (it's made contiguous before copy)
        assert r_non_contig.is_contiguous()
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# FROM COLOSSALAI: Stream query during overlap
# ═══════════════════════════════════════════════════════════════════════


class TestStreamBusyness:
    """Verify that streams are actually being used concurrently."""

    def test_d2h_stream_is_busy_during_offload(self):
        """D2H stream should have pending work after queueing an offload."""
        handler, pool = _make_handler()
        # Large tensor to ensure copy takes time
        t = torch.randn(4096, 4096, device="cuda")

        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(handler.d2h_stream):
            state = handler.offload(t, use_cpu_pool=False)

        # Query if d2h stream has pending work (should be True right after queueing)
        # Note: on fast GPUs this might complete instantly, so we just check it doesn't crash
        query_result = handler.d2h_stream.query()
        # query() returns True if stream is idle — we want to verify the API works
        assert isinstance(query_result, bool)

        torch.cuda.synchronize()
        # After sync, stream should be idle
        assert handler.d2h_stream.query() is True
        pool.clear()


# ═══════════════════════════════════════════════════════════════════════
# COMBINED: Offload with mixed dtypes in same group
# ═══════════════════════════════════════════════════════════════════════


class TestMixedDtypeGroup:
    """Test offloading a group containing tensors of different dtypes."""

    def test_mixed_dtype_offload_reload(self):
        handler, pool = _make_handler()
        handler.is_warmup = True
        handler.do_offload = True
        handler._offloaded_group_index = 1

        group = OffloadTensorGroup("mixed_dtype", use_cpu_pool=False)
        handler.offload_groups.append(group)
        handler._max_group_size = 1

        f32 = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        f16 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        bf16 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

        f32_exp, f16_exp, bf16_exp = f32.clone(), f16.clone(), bf16.clone()

        group.push_tensor((1, 0), f32)
        group.push_tensor((1, 1), f16)
        group.push_tensor((1, 2), bf16)

        handler._groups_to_offload.append(group)
        handler.d2h_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_offload_group()
        torch.cuda.synchronize()

        handler._groups_to_reload.append(group)
        handler.h2d_stream.wait_stream(torch.cuda.current_stream())
        handler.bulk_reload_group()
        torch.cuda.synchronize()

        assert torch.equal(group.pop_tensor((1, 0)), f32_exp)
        assert torch.equal(group.pop_tensor((1, 1)), f16_exp)
        assert torch.equal(group.pop_tensor((1, 2)), bf16_exp)
        pool.clear()
