#!/usr/bin/env python3
"""
Run ACTUAL torchtitan training with comprehensive logging to trace NaN.
This patches the model to log all intermediate tensors.
"""

import functools
import os
import sys

import torch
import torch.nn as nn

# Patch BEFORE importing torchtitan modules
_original_forward_methods = {}
_debug_rank = 0
_debug_step = 0
_nan_found = False
_log_file = None


def deep_check(name, tensor, rank, log_to_file=True):
    """Deep tensor check with all statistics."""
    global _nan_found

    if tensor is None:
        msg = f"[R{rank}][S{_debug_step}] {name}: None"
        print(msg)
        if _log_file:
            _log_file.write(msg + "\n")
        return True

    if not isinstance(tensor, torch.Tensor):
        msg = f"[R{rank}][S{_debug_step}] {name}: not a tensor ({type(tensor)})"
        print(msg)
        if _log_file:
            _log_file.write(msg + "\n")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        finite_tensor = tensor[finite_mask].float()
        min_val = finite_tensor.min().item()
        max_val = finite_tensor.max().item()
        mean_val = finite_tensor.mean().item()
        std_val = finite_tensor.std().item() if finite_tensor.numel() > 1 else 0
    else:
        min_val = max_val = mean_val = std_val = float("nan")

    status = (
        "OK"
        if not (has_nan or has_inf)
        else f"NaN:{nan_count}"
        if has_nan
        else f"Inf:{inf_count}"
    )

    msg = f"[R{rank}][S{_debug_step}] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f} [{status}]"

    if has_nan or has_inf:
        msg = "*** " + msg + " ***"
        _nan_found = True

    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()

    return not (has_nan or has_inf)


def make_logging_hook(module_name, hook_type):
    """Create a hook that logs tensor values."""

    def hook(module, input, output):
        global _debug_rank
        prefix = f"{module_name}.{hook_type}"

        # Log inputs
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    deep_check(f"{prefix}.input[{i}]", inp, _debug_rank)
        elif isinstance(input, torch.Tensor):
            deep_check(f"{prefix}.input", input, _debug_rank)

        # Log output
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    deep_check(f"{prefix}.output[{i}]", out, _debug_rank)
        elif isinstance(output, torch.Tensor):
            deep_check(f"{prefix}.output", output, _debug_rank)

    return hook


def register_all_hooks(model, prefix=""):
    """Register hooks on all modules."""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Register forward hook
        module.register_forward_hook(make_logging_hook(full_name, "fwd"))

        # Recursively register on children
        register_all_hooks(module, full_name)


def patch_flex_attention():
    """Patch flex_attention to log inputs/outputs."""
    from torchtitan.models.attention import FlexAttentionWrapper

    original_forward = FlexAttentionWrapper.forward

    @functools.wraps(original_forward)
    def patched_forward(self, q, k, v, *, block_mask, scale=None, return_lse=False):
        global _debug_rank, _debug_step

        print(
            f"\n[R{_debug_rank}][S{_debug_step}] ===== FlexAttentionWrapper.forward ====="
        )
        deep_check("FlexAttn.Q", q, _debug_rank)
        deep_check("FlexAttn.K", k, _debug_rank)
        deep_check("FlexAttn.V", v, _debug_rank)
        print(f"[R{_debug_rank}][S{_debug_step}] FlexAttn.scale: {scale}")
        print(
            f"[R{_debug_rank}][S{_debug_step}] FlexAttn.block_mask.shape: {block_mask.shape}"
        )

        result = original_forward(
            self, q, k, v, block_mask=block_mask, scale=scale, return_lse=return_lse
        )

        if isinstance(result, tuple):
            deep_check("FlexAttn.output[0]", result[0], _debug_rank)
            if len(result) > 1 and result[1] is not None:
                deep_check("FlexAttn.output[1](lse)", result[1], _debug_rank)
        else:
            deep_check("FlexAttn.output", result, _debug_rank)

        print(f"[R{_debug_rank}][S{_debug_step}] ===== FlexAttn END =====\n")
        return result

    FlexAttentionWrapper.forward = patched_forward


def patch_model_forward():
    """Patch the model's forward to log after each layer."""
    from torchtitan.models.deepseek_v3.model.model import TransformerBlock

    original_forward = TransformerBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, x, freqs_cis, attention_masks, *args, **kwargs):
        global _debug_rank, _debug_step

        deep_check(f"TransformerBlock[{self.layer_id}].input", x, _debug_rank)

        result = original_forward(self, x, freqs_cis, attention_masks, *args, **kwargs)

        deep_check(f"TransformerBlock[{self.layer_id}].output", result, _debug_rank)

        return result

    TransformerBlock.forward = patched_forward


def patch_attention():
    """Patch the Attention module."""
    from torchtitan.models.deepseek_v3.model.model import Attention

    original_forward = Attention.forward

    @functools.wraps(original_forward)
    def patched_forward(self, x, freqs_cis, attention_masks, *args, **kwargs):
        global _debug_rank, _debug_step

        print(f"\n[R{_debug_rank}][S{_debug_step}] ----- Attention.forward -----")
        deep_check("Attention.input_x", x, _debug_rank)

        result = original_forward(self, x, freqs_cis, attention_masks, *args, **kwargs)

        deep_check("Attention.output", result, _debug_rank)
        print(f"[R{_debug_rank}][S{_debug_step}] ----- Attention END -----\n")

        return result

    Attention.forward = patched_forward


def run_training():
    """Run actual torchtitan training with patches."""
    global _debug_rank, _debug_step, _log_file

    import torch.distributed as dist
    from torchtitan.train import Trainer

    # Apply patches
    print("Applying logging patches...")
    patch_flex_attention()
    patch_model_forward()
    patch_attention()

    # Initialize
    dist.init_process_group(backend="nccl")
    _debug_rank = dist.get_rank()

    # Open log file
    _log_file = open(
        f"/home/phuc/kimi_1t/torchtitan/debug_log_rank{_debug_rank}.txt", "w"
    )

    print(f"[R{_debug_rank}] Starting training with comprehensive logging...")

    # Create trainer with config - use the minimal CP=2 config
    config_path = "/home/phuc/kimi_1t/torchtitan/torchtitan/models/deepseek_v3/train_configs/debug_cp_nan/debug_flex_attn_cp2_minimal.toml"

    # Patch the training step to track step number
    original_train_step = None

    try:
        trainer = Trainer(config_path)

        # Manual training loop with logging
        trainer.build_components()

        # Manually step through training
        for step in range(1, 6):
            _debug_step = step
            print(f"\n{'#'*80}")
            print(
                f"[R{_debug_rank}] #################### TRAINING STEP {step} ####################"
            )
            print(f"{'#'*80}\n")

            try:
                # Get batch
                batch = next(trainer.data_iterator)
                deep_check("batch.input_ids", batch.input_ids, _debug_rank)
                deep_check("batch.labels", batch.labels, _debug_rank)

                # Forward/backward
                trainer.optimizer.zero_grad()

                # This calls the model forward
                with trainer.get_train_context():
                    loss = trainer.forward_step(batch)

                deep_check("loss", loss, _debug_rank)

                loss.backward()

                # Check gradients
                print(f"\n[R{_debug_rank}][S{step}] === CHECKING GRADIENTS ===")
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        ok = deep_check(f"grad.{name}", param.grad, _debug_rank)
                        if not ok:
                            print(
                                f"[R{_debug_rank}][S{step}] *** NaN GRADIENT: {name} ***"
                            )

                trainer.optimizer.step()

                loss_val = loss.item() if torch.isfinite(loss) else float("nan")
                print(f"\n[R{_debug_rank}][S{step}] === STEP {step} SUMMARY ===")
                print(f"[R{_debug_rank}][S{step}] Loss: {loss_val}")

                if not torch.isfinite(loss):
                    print(f"[R{_debug_rank}][S{step}] *** NaN LOSS DETECTED ***")
                    break

            except Exception as e:
                print(f"[R{_debug_rank}][S{step}] ERROR: {e}")
                import traceback

                traceback.print_exc()
                break

    except Exception as e:
        print(f"[R{_debug_rank}] Trainer initialization error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if _log_file:
            _log_file.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_training()
