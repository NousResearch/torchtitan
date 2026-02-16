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
        trainer_name: str = "torchtitan",
    ) -> bool:
        """
        Register this trainer with Atropos API.

        Args:
            batch_size: Batch size to request
            max_token_len: Maximum token length
            num_steps: Expected number of training steps
            trainer_name: Name identifier for this trainer

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
                    "trainer_name": trainer_name,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.trainer_id = data.get("trainer_id")
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
                    f"{self.api_url}/get_batch",
                    params={"trainer_id": self.trainer_id} if self.trainer_id else {},
                    timeout=self.timeout,
                )

                if response.status_code == 204:
                    # No batch available
                    if not block:
                        return None
                    time.sleep(self.poll_interval)
                    continue

                if response.status_code == 410:
                    # Training complete
                    print("[AtroposClient] Training complete signal received")
                    return None

                response.raise_for_status()
                data = response.json()

                return AtroposBatch(
                    tokens=data.get("tokens", []),
                    masks=data.get("masks", []),
                    scores=data.get("scores", []),
                    advantages=data.get("advantages"),
                    ref_logprobs=data.get("ref_logprobs"),
                    messages=data.get("messages"),
                    generation_params=data.get("generation_params"),
                    inference_logprobs=data.get("inference_logprobs"),
                    group_overrides=data.get("group_overrides"),
                    overrides=data.get("overrides"),
                    images=data.get("images"),
                    onpolicydistill_logprobs=data.get("onpolicydistill_logprobs"),
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
        Report training step completion to Atropos.

        Args:
            step: Current training step
            metrics: Optional training metrics (loss, reward, etc.)

        Returns:
            True if report successful
        """
        try:
            response = requests.post(
                f"{self.api_url}/report_step",
                json={
                    "trainer_id": self.trainer_id,
                    "step": step,
                    "metrics": metrics or {},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"[AtroposClient] Error reporting step: {e}")
            return False

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
