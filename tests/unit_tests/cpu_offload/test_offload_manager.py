# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for OffloadManager — singleton orchestrator."""

import pytest
import torch

from torchtitan.distributed.cpu_offload.offload_manager import OffloadManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


class TestSingleton:

    def test_get_instance_returns_same_object(self):
        m1 = OffloadManager.get_instance()
        m2 = OffloadManager.get_instance()
        assert m1 is m2

    def test_reset_instance_creates_new(self):
        m1 = OffloadManager.get_instance()
        OffloadManager.reset_instance()
        m2 = OffloadManager.get_instance()
        assert m1 is not m2

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestStreams:

    def test_has_dedicated_streams(self):
        mgr = OffloadManager.get_instance()
        assert isinstance(mgr.d2h_stream, torch.cuda.Stream)
        assert isinstance(mgr.h2d_stream, torch.cuda.Stream)
        # Streams should be different from default
        assert mgr.d2h_stream != torch.cuda.default_stream()
        assert mgr.h2d_stream != torch.cuda.default_stream()

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestChunkInitialization:

    def test_init_chunk_handler_creates_chunk(self):
        mgr = OffloadManager.get_instance()
        mgr.init_chunk_handler(vp_size=1, vp_stage=0, min_offloaded_tensor_size=1)
        assert mgr.cur_forward_chunk() is not None

    def test_init_multiple_chunks(self):
        mgr = OffloadManager.get_instance()
        mgr.init_chunk_handler(vp_size=1, vp_stage=0)
        c1 = mgr.cur_forward_chunk()
        # Simulate next microbatch — but since warmup only creates on first call
        # and subsequent calls are no-ops, we test the first chunk exists
        assert c1 is not None
        assert c1.is_warmup is True

    def test_init_with_vpp(self):
        mgr = OffloadManager.get_instance()
        mgr.init_chunk_handler(vp_size=2, vp_stage=0)
        c0 = mgr.cur_forward_chunk()
        assert c0 is not None
        assert c0.vpp_rank == 0

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestDisableEnable:

    def test_disable_sets_flag(self):
        mgr = OffloadManager.get_instance()
        mgr.init_chunk_handler(vp_size=1, vp_stage=0)
        mgr.disable_offload()
        assert mgr.do_offload is False
        chunk = mgr.cur_forward_chunk()
        assert chunk.do_offload is False

    def test_enable_restores_flag(self):
        mgr = OffloadManager.get_instance()
        mgr.init_chunk_handler(vp_size=1, vp_stage=0)
        mgr.disable_offload()
        mgr.enable_offload()
        assert mgr.do_offload is True
        chunk = mgr.cur_forward_chunk()
        assert chunk.do_offload is True

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestMarkNotOffloadable:

    def test_marks_tensor(self):
        mgr = OffloadManager.get_instance()
        t = torch.randn(10, device="cuda")
        mgr.mark_not_offloadable(t)
        assert hasattr(t, "offloading_activation")
        assert t.offloading_activation is False

    def test_none_tensor_no_error(self):
        mgr = OffloadManager.get_instance()
        mgr.mark_not_offloadable(None)  # should not raise

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestContextManager:

    def test_enter_exit_without_chunk(self):
        """Context manager should be a no-op when no chunk is active."""
        mgr = OffloadManager.get_instance()
        # No init_chunk_handler called, so _cur_forward_chunk is None
        mgr.__enter__()  # should not raise
        mgr.__exit__()   # should not raise

    def teardown_method(self):
        OffloadManager._INSTANCE = None


class TestDelayedOffload:

    def test_push_and_flush(self):
        mgr = OffloadManager.get_instance()
        calls = []

        def hook(tensors):
            calls.append(("hook", tensors))

        mgr.push_offload_groups(hook, [1, 2])
        mgr.push_offload_groups(hook, [3, 4])
        assert len(calls) == 0

        mgr.flush_delayed_groups()
        # Flushed in reverse order
        assert len(calls) == 2
        assert calls[0] == ("hook", [3, 4])
        assert calls[1] == ("hook", [1, 2])

    def teardown_method(self):
        OffloadManager._INSTANCE = None
