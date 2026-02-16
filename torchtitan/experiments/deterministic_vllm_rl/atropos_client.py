# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atropos API client for fetching training batches.

This client connects to the Atropos API server to:
1. Register as a trainer
2. Poll for and receive scored training batches
3. Report training progress

The batches may include `onpolicydistill_logprobs` from a teacher model
for on-policy distillation.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class AtroposBatch:
    """Container for a training batch from Atropos API."""

    tokens: list[list[int]]
    masks: list[list[int]]
    scores: list[float]
    advantages: list[list[float]] | None = None
    ref_logprobs: list[list[float]] | None = None
    messages: list[list[dict]] | None = None
    generation_params: dict[str, Any] | None = None
    inference_logprobs: list[list[float]] | None = None
    group_overrides: dict | None = None
    overrides: list[dict] | None = None
    images: Any = None
    # On-policy distillation: top-K logprobs from teacher model
    # Structure: [sequence][position][top_k] = [token_id, logprob]
    onpolicydistill_logprobs: list[list[list[list]]] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for training functions."""
        return {
            "tokens": self.tokens,
            "masks": self.masks,
            "scores": self.scores,
            "advantages": self.advantages,
            "ref_logprobs": self.ref_logprobs,
            "messages": self.messages,
            "generation_params": self.generation_params,
            "inference_logprobs": self.inference_logprobs,
            "group_overrides": self.group_overrides,
            "overrides": self.overrides,
            "images": self.images,
            "onpolicydistill_logprobs": self.onpolicydistill_logprobs,
        }

    def __len__(self) -> int:
        return len(self.tokens)


class AtroposClient:
    """
    Client for communicating with Atropos API server.

    Usage:
        client = AtroposClient("http://localhost:8000")
        client.register(batch_size=64, max_token_len=2048, num_steps=100)

        for step in range(num_steps):
            batch = client.get_batch(block=True)
            if batch is None:
                break

            # Train with batch
            loss = train_step(batch)

            client.report_step(step, {"loss": loss})
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        poll_interval: float = 1.0,
        timeout: float = 30.0,
    ):
        """
        Initialize Atropos client.

        Args:
            api_url: Base URL of Atropos API server
            poll_interval: Seconds between polls when waiting for batch
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.trainer_id: str | None = None
        self._registered = False

    def register(
        self,
        batch_size: int = 64,
        max_token_len: int = 2048,
        num_steps: int = 100,
        wandb_group: str | None = None,
        wandb_project: str | None = None,
        checkpoint_dir: str = "./checkpoints",
        save_checkpoint_interval: int = 100,
        starting_step: int = 0,
    ) -> bool:
        """
        Register this trainer with Atropos API.

        Args:
            batch_size: Batch size to request
            max_token_len: Maximum token length
            num_steps: Expected number of training steps
            wandb_group: Wandb group name (auto-generated if None)
            wandb_project: Wandb project name (from env var if None)
            checkpoint_dir: Directory for saving checkpoints
            save_checkpoint_interval: Steps between checkpoint saves
            starting_step: Starting training step

        Returns:
            True if registration successful
        """
        import os
        import time

        # Use env vars or generate defaults
        if wandb_group is None:
            wandb_group = f"torchtitan-{int(time.time())}"
        if wandb_project is None:
            wandb_project = os.environ.get("WANDB_PROJECT", "torchtitan-rl")

        try:
            response = requests.post(
                f"{self.api_url}/register",
                json={
                    "wandb_group": wandb_group,
                    "wandb_project": wandb_project,
                    "batch_size": batch_size,
                    "max_token_len": max_token_len,
                    "checkpoint_dir": checkpoint_dir,
                    "save_checkpoint_interval": save_checkpoint_interval,
                    "starting_step": starting_step,
                    "num_steps": num_steps,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.trainer_id = data.get("uuid")
            self._registered = True
            print(f"[AtroposClient] Registered with ID: {self.trainer_id}")
            return True
        except requests.RequestException as e:
            print(f"[AtroposClient] Registration failed: {e}")
            return False

    def get_batch(self, block: bool = True) -> AtroposBatch | None:
        """
        Get next training batch from Atropos.

        Args:
            block: If True, poll until batch available. If False, return None immediately.

        Returns:
            AtroposBatch or None if no batch available (non-blocking) or training complete
        """
        while True:
            try:
                response = requests.get(
                    f"{self.api_url}/batch",
                    timeout=self.timeout,
                )

                response.raise_for_status()
                data = response.json()

                # Atropos returns {"batch": <batch_data or None>}
                batch_data = data.get("batch")

                if batch_data is None:
                    # No batch available yet
                    if not block:
                        return None
                    time.sleep(self.poll_interval)
                    continue

                # batch_data is a list of scored data groups
                # Merge them into a single AtroposBatch
                all_tokens = []
                all_masks = []
                all_scores = []
                all_advantages = []
                all_ref_logprobs = []
                all_messages = []
                all_distill_logprobs = []

                for group in batch_data:
                    all_tokens.extend(group.get("tokens", []))
                    all_masks.extend(group.get("masks", []))
                    all_scores.extend(group.get("scores", []))
                    if group.get("advantages"):
                        all_advantages.extend(group.get("advantages"))
                    if group.get("ref_logprobs"):
                        all_ref_logprobs.extend(group.get("ref_logprobs"))
                    if group.get("messages"):
                        all_messages.extend(group.get("messages"))
                    if group.get("onpolicydistill_logprobs"):
                        all_distill_logprobs.extend(group.get("onpolicydistill_logprobs"))

                return AtroposBatch(
                    tokens=all_tokens,
                    masks=all_masks,
                    scores=all_scores,
                    advantages=all_advantages if all_advantages else None,
                    ref_logprobs=all_ref_logprobs if all_ref_logprobs else None,
                    messages=all_messages if all_messages else None,
                    onpolicydistill_logprobs=all_distill_logprobs if all_distill_logprobs else None,
                )

            except requests.RequestException as e:
                print(f"[AtroposClient] Error fetching batch: {e}")
                if not block:
                    return None
                time.sleep(self.poll_interval)

    def report_step(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        """
        Report training step completion (local logging only).

        The Atropos API tracks steps internally via batch fetches.
        This method is for local logging/tracking purposes.

        Args:
            step: Current training step
            metrics: Optional training metrics (loss, reward, etc.)

        Returns:
            True always (no remote call)
        """
        # Atropos tracks steps internally - no need to report
        # Just log locally if desired
        return True

    def get_status(self) -> dict:
        """Get current training status from Atropos API."""
        try:
            response = requests.get(
                f"{self.api_url}/status",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[AtroposClient] Error getting status: {e}")
            return {"current_step": -1, "queue_size": -1}

    def close(self) -> None:
        """Unregister from Atropos API."""
        if self._registered and self.trainer_id:
            try:
                requests.post(
                    f"{self.api_url}/unregister",
                    json={"trainer_id": self.trainer_id},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                pass  # Best effort
