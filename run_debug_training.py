#!/usr/bin/env python3
"""
Run torchtitan training with debug hooks enabled.
This traces all tensor operations to find where NaN originates.
"""

import os
import sys

import torch
import torch.distributed as dist

# Add torchtitan to path
sys.path.insert(0, "/home/phuc/kimi_1t/torchtitan")

# Enable debug hooks BEFORE importing torchtitan
from debug_hooks import check_tensor, enable_all_patches, set_debug_rank, set_debug_step

enable_all_patches()

# Now import torchtitan training
from torchtitan.train import Trainer


def main():
    # Parse config from command line args
    if len(sys.argv) < 3:
        print("Usage: run_debug_training.py --config <config_path>")
        sys.exit(1)

    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if config_path is None:
        print("Error: --config argument required")
        sys.exit(1)

    print(f"Starting debug training with config: {config_path}")

    # Initialize trainer
    trainer = Trainer(config_path)

    # Get rank after trainer init
    if dist.is_initialized():
        set_debug_rank(dist.get_rank())
        rank = dist.get_rank()
    else:
        rank = 0

    print(f"[R{rank}] Trainer initialized, starting training loop...")

    # Run training with step tracking
    try:
        for step in range(1, trainer.job_config.training.steps + 1):
            set_debug_step(step)
            print(f"\n{'#'*80}")
            print(f"[R{rank}] #################### STEP {step} ####################")
            print(f"{'#'*80}\n")

            # Get batch
            batch = trainer.data_loader.get_next_batch()

            check_tensor("batch.input_ids", batch.input_ids, log_all=True)
            check_tensor("batch.labels", batch.labels, log_all=True)

            # Forward pass
            trainer.optimizer.zero_grad()

            with trainer.get_train_context():
                loss, metrics = trainer.train_step(batch)

            check_tensor(
                "loss", loss.unsqueeze(0) if loss.dim() == 0 else loss, log_all=True
            )

            # Backward pass
            loss.backward()

            # Check gradients
            print(f"\n[R{rank}][S{step}] === GRADIENT CHECK ===")
            nan_grads = []
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    ok = check_tensor(f"grad.{name}", param.grad, log_all=False)
                    if not ok:
                        nan_grads.append(name)

            if nan_grads:
                print(f"[R{rank}][S{step}] *** NaN GRADIENTS IN: {nan_grads} ***")

            # Optimizer step
            trainer.optimizer.step()
            trainer.lr_scheduler.step()

            loss_val = loss.item() if torch.isfinite(loss) else float("nan")
            print(f"\n[R{rank}][S{step}] === STEP {step} DONE: loss={loss_val:.4f} ===")

            if not torch.isfinite(loss):
                print(f"[R{rank}][S{step}] *** NaN/Inf LOSS - STOPPING ***")
                break

    except Exception as e:
        print(f"[R{rank}] Training error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n[R{rank}] Debug training complete")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
