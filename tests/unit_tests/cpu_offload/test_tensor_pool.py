# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for TensorPool — pinned CPU memory pool."""

import pytest
import torch

from torchtitan.distributed.cpu_offload.tensor_pool import TensorPool


class TestTensorPoolBasic:
    """Basic allocation and free tests (CPU-only, no GPU needed)."""

    def test_allocate_returns_correct_shape_and_dtype(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t = pool.allocate((4, 8), dtype=torch.float32)
        assert t.shape == (4, 8)
        assert t.dtype == torch.float32
        assert t.device == torch.device("cpu")
        pool.clear()

    def test_allocate_multiple_dtypes(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t_f32 = pool.allocate((2, 3), dtype=torch.float32)
        t_f16 = pool.allocate((2, 3), dtype=torch.float16)
        t_bf16 = pool.allocate((2, 3), dtype=torch.bfloat16)
        assert t_f32.dtype == torch.float32
        assert t_f16.dtype == torch.float16
        assert t_bf16.dtype == torch.bfloat16
        pool.clear()

    def test_free_returns_tensor_to_pool(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t1 = pool.allocate((4, 8), dtype=torch.float32)
        assert pool.stats["current_in_use"] == 1
        pool.free(t1)
        assert pool.stats["current_in_use"] == 0
        pool.clear()

    def test_pool_reuse_after_free(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t1 = pool.allocate((4, 8), dtype=torch.float32)
        t1_id = id(t1)
        pool.free(t1)
        t2 = pool.allocate((4, 8), dtype=torch.float32)
        # Should reuse the same tensor
        assert id(t2) == t1_id
        assert pool.stats["pool_hits"] == 1
        assert pool.stats["pool_misses"] == 1  # first alloc was a miss
        pool.clear()

    def test_pool_miss_when_different_shape(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t1 = pool.allocate((4, 8), dtype=torch.float32)
        pool.free(t1)
        t2 = pool.allocate((8, 4), dtype=torch.float32)
        # Different shape -> different pool -> miss
        assert id(t2) != id(t1)
        assert pool.stats["pool_misses"] == 2
        pool.free(t2)
        pool.clear()

    def test_pool_miss_when_different_dtype(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t1 = pool.allocate((4, 8), dtype=torch.float32)
        pool.free(t1)
        t2 = pool.allocate((4, 8), dtype=torch.float16)
        assert pool.stats["pool_misses"] == 2
        pool.free(t2)
        pool.clear()

    def test_free_unknown_tensor_raises(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        rogue = torch.empty(4, 8)
        with pytest.raises(ValueError, match="No pool"):
            pool.free(rogue)

    def test_free_wrong_pool_tensor_raises(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        _ = pool.allocate((4, 8), dtype=torch.float32)
        rogue = torch.empty(4, 8)
        with pytest.raises(ValueError, match="does not belong"):
            pool.free(rogue)
        pool.clear()

    def test_reset_makes_all_available(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        t1 = pool.allocate((4, 8), dtype=torch.float32)
        t2 = pool.allocate((4, 8), dtype=torch.float32)
        assert pool.stats["current_in_use"] == 2
        pool.reset()
        assert pool.stats["current_in_use"] == 0
        # After reset, both should be reusable
        a = pool.allocate((4, 8), dtype=torch.float32)
        b = pool.allocate((4, 8), dtype=torch.float32)
        assert {id(a), id(b)} == {id(t1), id(t2)}
        pool.clear()

    def test_clear_releases_everything(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        _ = pool.allocate((4, 8), dtype=torch.float32)
        pool.clear()
        assert pool.stats["current_in_use"] == 0
        # After clear, allocating same shape creates a new tensor
        t2 = pool.allocate((4, 8), dtype=torch.float32)
        assert pool.stats["pool_misses"] == 2  # both were misses
        pool.clear()

    def test_many_allocations_and_frees(self):
        pool = TensorPool(device="cpu", pin_memory=False)
        tensors = [pool.allocate((16, 32), dtype=torch.float32) for _ in range(100)]
        assert pool.stats["total_allocated"] == 100
        for t in tensors:
            pool.free(t)
        assert pool.stats["current_in_use"] == 0
        # Reallocate — all should be hits
        tensors2 = [pool.allocate((16, 32), dtype=torch.float32) for _ in range(100)]
        assert pool.stats["pool_hits"] == 100
        for t in tensors2:
            pool.free(t)
        pool.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTensorPoolPinned:
    """Tests with pinned memory on CUDA systems."""

    def test_pinned_memory_allocation(self):
        pool = TensorPool(device="cpu", pin_memory=True)
        t = pool.allocate((128, 256), dtype=torch.float32)
        assert t.is_pinned()
        assert t.device == torch.device("cpu")
        pool.free(t)
        pool.clear()

    def test_pinned_pool_reuse_stays_pinned(self):
        pool = TensorPool(device="cpu", pin_memory=True)
        t1 = pool.allocate((64,), dtype=torch.float16)
        assert t1.is_pinned()
        pool.free(t1)
        t2 = pool.allocate((64,), dtype=torch.float16)
        assert t2.is_pinned()
        assert id(t2) == id(t1)
        pool.clear()
