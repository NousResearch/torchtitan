# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.tools.logging import logger

class NumericalHealthMonitor:
    """
    Monitor to track numerical stability (gradients, rewards) during GRPO training.
    Provides automated logging and health status checks.
    """
    def __init__(self, reward_std_threshold=1e-6):
        self.reward_std_threshold = reward_std_threshold
        logger.info("NumericalHealthMonitor initialized")

    @torch.no_grad()
    def check(self, step, rewards=None, model_parts=None):
        status = {"healthy": True, "issues": []}
        
        # 1. Gradient Norm Check (if model provided)
        if model_parts is not None:
            total_grad_norm = 0.0
            for part in model_parts:
                for p in part.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if not torch.isfinite(torch.tensor(total_grad_norm)):
                status["healthy"] = False
                status["issues"].append("Non-finite gradients detected")
                logger.error(f"Step {step}: CRITICAL - Non-finite gradients in model!")

        # 2. Reward Distribution Check
        if rewards is not None and rewards.numel() > 1:
            r_std = rewards.std().item()
            r_max = rewards.max().item()
            r_min = rewards.min().item()
            
            if not torch.isfinite(rewards).all():
                status["healthy"] = False
                status["issues"].append("Non-finite rewards detected")
                logger.error(f"Step {step}: CRITICAL - Non-finite rewards in batch!")
            
            if r_std < self.reward_std_threshold:
                status["issues"].append(f"Collapsed reward std: {r_std:.8f}")
                logger.warning(f"Step {step}: Warning - Collapsed reward distribution (std={r_std:.8f})")
                
            if abs(r_max) > 1e4 or abs(r_min) > 1e4:
                status["issues"].append(f"Extreme rewards: [{r_min:.2f}, {r_max:.2f}]")
                logger.warning(f"Step {step}: Warning - Extreme rewards detected")

        return status
