# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atropos API client for fetching training batches.

This module provides a simple interface for TorchTitan RL training loops
to fetch batches from an Atropos environment server.

Usage:
    from torchtitan.experiments.rl.atropos_client import AtroposClient
    
    client = AtroposClient("http://localhost:8000")
    client.register(batch_size=64, max_token_len=2048, num_steps=1000)
    
    while training:
        batch = client.get_batch()  # Blocks until batch available
        if batch is None:
            break  # Training complete
        
        # batch contains:
        #   - tokens: List[List[int]]
        #   - masks: List[List[int]]  
        #   - scores: List[float]
        #   - onpolicydistill_logprobs: Optional[List] (if distillation enabled)
"""

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class AtroposBatch:
    """A batch of training data from Atropos API."""
    
    tokens: list[list[int]]
    masks: list[list[int]]
    scores: list[float]
    advantages: list[float] | None = None
    messages: list[list[dict]] | None = None
    onpolicydistill_logprobs: list | None = None
    
    @classmethod
    def from_api_response(cls, data: dict) -> "AtroposBatch":
        """Create batch from API response dict."""
        return cls(
            tokens=data.get("tokens", []),
            masks=data.get("masks", []),
            scores=data.get("scores", []),
            advantages=data.get("advantages"),
            messages=data.get("messages"),
            onpolicydistill_logprobs=data.get("onpolicydistill_logprobs"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for use with training functions."""
        return {
            "tokens": self.tokens,
            "masks": self.masks,
            "scores": self.scores,
            "advantages": self.advantages,
            "messages": self.messages,
            "onpolicydistill_logprobs": self.onpolicydistill_logprobs,
        }


class AtroposClient:
    """
    Client for fetching training batches from Atropos API.
    
    Args:
        api_url: Base URL of Atropos API (e.g., "http://localhost:8000")
        poll_interval: Seconds to wait between polling for batches
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        poll_interval: float = 1.0,
        timeout: float = 30.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._registered = False
    
    def register(
        self,
        batch_size: int,
        max_token_len: int,
        num_steps: int,
        starting_step: int = 0,
        checkpoint_dir: str | None = None,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
    ) -> bool:
        """
        Register trainer with Atropos API.
        
        Args:
            batch_size: Number of samples per batch
            max_token_len: Maximum sequence length
            num_steps: Total training steps
            starting_step: Step to resume from
            checkpoint_dir: Directory for checkpoints
            wandb_project: Weights & Biases project name
            wandb_group: Weights & Biases group name
            
        Returns:
            True if registration successful
        """
        try:
            response = requests.post(
                f"{self.api_url}/register",
                json={
                    "batch_size": batch_size,
                    "max_token_len": max_token_len,
                    "num_steps": num_steps,
                    "starting_step": starting_step,
                    "checkpoint_dir": checkpoint_dir,
                    "wandb_project": wandb_project,
                    "wandb_group": wandb_group,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            self._registered = True
            print(f"Registered with Atropos API at {self.api_url}")
            return True
        except Exception as e:
            print(f"Failed to register with Atropos API: {e}")
            return False
    
    def get_batch(self, block: bool = True) -> AtroposBatch | None:
        """
        Fetch next training batch from API.
        
        Args:
            block: If True, poll until batch available. If False, return None immediately.
            
        Returns:
            AtroposBatch if available, None if training complete or no batch ready
        """
        while True:
            try:
                response = requests.get(
                    f"{self.api_url}/batch",
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                
                # Check if batch is available
                if data.get("batch") is not None:
                    # Flatten batch groups into single batch
                    batches = []
                    for group in data["batch"]:
                        batches.append(AtroposBatch.from_api_response(group))
                    
                    # Merge all groups (simple case: return first)
                    # In production, you'd want to handle multiple groups properly
                    if batches:
                        return batches[0]
                
                # No batch available
                if not block:
                    return None
                    
                # Poll again after interval
                time.sleep(self.poll_interval)
                
            except requests.exceptions.ConnectionError:
                if not block:
                    return None
                print(f"Waiting for Atropos API at {self.api_url}...")
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"Error fetching batch: {e}")
                if not block:
                    return None
                time.sleep(self.poll_interval)
    
    def report_step(self, step: int, metrics: dict | None = None) -> bool:
        """
        Report training progress to API.
        
        Args:
            step: Current training step
            metrics: Optional metrics dict to report
            
        Returns:
            True if report successful
        """
        try:
            response = requests.post(
                f"{self.api_url}/step",
                json={"step": step, "metrics": metrics or {}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to report step: {e}")
            return False
