# Copyright (c) 2025 Nous Research. All rights reserved.

"""Tests for utility functions."""

import pytest
import torch

from torchtitan.distributed.cpu_offload.utils import (
    debug_rank,
    is_graph_capturing,
    print_offload_summary_table,
)


class TestDebugRank:

    def test_debug_disabled_no_output(self, capsys):
        import torchtitan.distributed.cpu_offload.utils as utils
        original = utils.DEBUG
        utils.DEBUG = False
        debug_rank("should not print")
        captured = capsys.readouterr()
        assert captured.out == ""
        utils.DEBUG = original

    def test_debug_enabled_prints(self, capsys):
        import torchtitan.distributed.cpu_offload.utils as utils
        original = utils.DEBUG
        utils.DEBUG = True
        debug_rank("hello debug")
        captured = capsys.readouterr()
        assert "hello debug" in captured.out
        utils.DEBUG = original


class TestIsGraphCapturing:

    def test_returns_bool(self):
        result = is_graph_capturing()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_not_capturing_normally(self):
        assert is_graph_capturing() is False


class TestPrintOffloadSummary:

    def test_local_summary(self, capsys):
        data = {"expert_fc1": 1024 * 1024 * 50, "moe_act": 1024 * 1024 * 30}
        print_offload_summary_table(data, distributed=False)
        captured = capsys.readouterr()
        assert "Activation Offload Summary" in captured.out
        assert "expert_fc1" in captured.out
        assert "moe_act" in captured.out

    def test_empty_summary(self, capsys):
        print_offload_summary_table({}, distributed=False)
        captured = capsys.readouterr()
        assert captured.out == ""  # empty dict -> no output
