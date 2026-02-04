"""
Debug hooks for tracing NaN in training.
Import this before running torchtitan training to enable comprehensive logging.
"""

import functools

import torch

_debug_rank = 0
_debug_step = [0]  # Use list to be mutable from inner functions
_log_enabled = True


def set_debug_rank(rank):
    global _debug_rank
    _debug_rank = rank


def set_debug_step(step):
    _debug_step[0] = step


def check_tensor(name, tensor, log_all=False):
    """Check tensor for NaN/Inf and optionally log all values."""
    global _debug_rank, _log_enabled

    if not _log_enabled:
        return True

    if tensor is None:
        return True

    if not isinstance(tensor, torch.Tensor):
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf or log_all:
        nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
        inf_count = torch.isinf(tensor).sum().item() if has_inf else 0

        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            min_val = tensor[finite_mask].float().min().item()
            max_val = tensor[finite_mask].float().max().item()
            mean_val = tensor[finite_mask].float().mean().item()
        else:
            min_val = max_val = mean_val = float("nan")

        status = (
            "OK"
            if not (has_nan or has_inf)
            else f"NaN:{nan_count}"
            if has_nan
            else f"Inf:{inf_count}"
        )
        print(
            f"[R{_debug_rank}][S{_debug_step[0]}] {name}: shape={tuple(tensor.shape)}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f} [{status}]"
        )

    return not (has_nan or has_inf)


def patch_flex_attention_wrapper():
    """Patch FlexAttentionWrapper to log inputs/outputs."""
    from torchtitan.models.attention import FlexAttentionWrapper

    original_forward = FlexAttentionWrapper.forward

    @functools.wraps(original_forward)
    def traced_forward(self, q, k, v, *, block_mask, scale=None, return_lse=False):
        global _debug_rank

        print(f"\n[R{_debug_rank}][S{_debug_step[0]}] ===== FlexAttentionWrapper =====")
        check_tensor("FlexAttn.Q", q, log_all=True)
        check_tensor("FlexAttn.K", k, log_all=True)
        check_tensor("FlexAttn.V", v, log_all=True)
        print(
            f"[R{_debug_rank}][S{_debug_step[0]}] scale={scale}, block_mask.shape={block_mask.shape}"
        )

        result = original_forward(
            self, q, k, v, block_mask=block_mask, scale=scale, return_lse=return_lse
        )

        if isinstance(result, tuple):
            check_tensor("FlexAttn.out[0]", result[0], log_all=True)
            if len(result) > 1 and result[1] is not None:
                check_tensor("FlexAttn.out[1](lse)", result[1], log_all=True)
        else:
            check_tensor("FlexAttn.output", result, log_all=True)

        print(f"[R{_debug_rank}][S{_debug_step[0]}] ===== FlexAttn END =====\n")
        return result

    FlexAttentionWrapper.forward = traced_forward
    print("Patched FlexAttentionWrapper.forward")


def patch_deepseek_attention():
    """Patch DeepSeek Attention module."""
    try:
        from torchtitan.models.deepseek_v3.model.model import Attention

        original_forward = Attention.forward

        @functools.wraps(original_forward)
        def traced_forward(self, x, freqs_cis, attention_masks, *args, **kwargs):
            check_tensor("Attention.input_x", x, log_all=True)

            result = original_forward(
                self, x, freqs_cis, attention_masks, *args, **kwargs
            )

            check_tensor("Attention.output", result, log_all=True)
            return result

        Attention.forward = traced_forward
        print("Patched DeepSeek Attention.forward")
    except ImportError:
        print("Could not patch DeepSeek Attention (not imported)")


def enable_all_patches():
    """Enable all debug patches."""
    patch_flex_attention_wrapper()
    patch_deepseek_attention()
    print("All debug patches enabled")
