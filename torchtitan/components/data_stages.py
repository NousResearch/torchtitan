# Copyright (c) Nous Research.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-stage data training support.

This module provides DataStageManager and StageAwareDataloader for training
with different data mixtures at different stages of training, similar to
approaches used in Qwen3, DeepSeek-V3, and Llama 3.

Data stages are OPTIONAL. If no [[training.data_stages]] are defined, a single
stage is auto-created from [training] data fields for backward compatibility.
When stages ARE defined, they override [training] data fields completely.

Example usage (multi-stage):
    [[training.data_stages]]
    name = "general"
    start_step = 0
    end_step = 100000
    dataset_type = "nanoset"
    dataset_folders = ["/data/web", "/data/books", "/data/code"]
    dataset_weights = [0.7, 0.2, 0.1]
    seq_len = 4096

    [[training.data_stages]]
    name = "reasoning"
    start_step = 100000
    dataset_type = "nanoset"
    dataset_folders = ["/data/web", "/data/books", "/data/code"]
    dataset_weights = [0.3, 0.35, 0.35]
    seq_len = 4096

Example usage (single-stage, backward compatible):
    [training]
    dataset_type = "huggingface"
    dataset = "c4_test"
    seq_len = 4096
    # No [[training.data_stages]] needed - auto-created internally
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config.job_config import DataStage, JobConfig
from torchtitan.tools.logging import logger


@dataclass
class EffectiveStageConfig:
    """Resolved stage config. Each stage must define all required fields."""

    dataset: str | None
    dataset_path: str | None
    dataset_type: str
    dataset_folders: list[str]
    dataset_weights: list[float] | None
    dataset_random_seed: int
    seq_len: int


class DataStageManager:
    """Manages data stage transitions during training.

    Tracks current stage based on training step, handles stage transitions,
    and builds dataloaders with stage-specific configurations.
    """

    def __init__(
        self,
        job_config: JobConfig,
        build_dataloader_fn: Callable,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: Any,
    ):
        self.job_config = job_config
        self.build_dataloader_fn = build_dataloader_fn
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.tokenizer = tokenizer

        # Convert dicts to DataStage objects if needed (TOML parser returns dicts)
        raw_stages = job_config.training.data_stages
        self.stages: list[DataStage] = []
        for stage in raw_stages:
            if isinstance(stage, dict):
                self.stages.append(DataStage(**stage))
            else:
                self.stages.append(stage)

        training = job_config.training

        # Case 1: No stages defined - auto-create from [training] (backward compat)
        if not self.stages:
            auto_stage = DataStage(
                name="default",
                start_step=0,
                end_step=None,
                dataset=training.dataset,
                dataset_path=training.dataset_path,
                dataset_type=training.dataset_type,
                dataset_folders=training.dataset_folders,
                dataset_weights=training.dataset_weights,
                dataset_random_seed=training.dataset_random_seed,
                seq_len=training.seq_len,
            )
            self.stages.append(auto_stage)
            logger.info(
                "No [[training.data_stages]] defined. "
                "Auto-created single stage from [training] config."
            )
        else:
            # Sort stages by start_step first to find the earliest
            self.stages.sort(key=lambda s: s.start_step)
            first_stage_start = self.stages[0].start_step

            # Check if [training] has non-default data fields
            training_has_data = self._has_non_default_training_data(training)

            if first_stage_start == 0:
                # Case 2: Stages cover from step 0 - warn if [training] has non-default data
                if training_has_data:
                    raise ValueError(
                        "Cannot define data fields in both [training] and [[training.data_stages]].\n"
                        "Your [[training.data_stages]] starts at step 0, so it fully covers training.\n"
                        "Please either:\n"
                        "  1. Remove data fields from [training] and use [[training.data_stages]] only, OR\n"
                        "  2. Remove [[training.data_stages]] and use [training] only\n\n"
                        "Detected non-default [training] data fields:\n"
                        f"{self._format_training_data_fields(training)}"
                    )
            else:
                # Case 3: Stages start after step 0 - auto-create stage from [training]
                # for the gap (ablation/resume use case)
                # Always use [training] values for the gap, even if they are defaults
                auto_stage = DataStage(
                    name="default",
                    start_step=0,
                    end_step=first_stage_start,
                    dataset=training.dataset,
                    dataset_path=training.dataset_path,
                    dataset_type=training.dataset_type,
                    dataset_folders=training.dataset_folders,
                    dataset_weights=training.dataset_weights,
                    dataset_random_seed=training.dataset_random_seed,
                    seq_len=training.seq_len,
                )
                self.stages.insert(0, auto_stage)
                logger.info(
                    f"[[training.data_stages]] starts at step {first_stage_start}. "
                    f"Auto-created 'default' stage from [training] for steps 0-{first_stage_start}."
                )

        self._current_stage_idx = 0

        # Sort stages by start_step for consistent ordering
        self.stages.sort(key=lambda s: s.start_step)
        self._validate_stages()
        self._log_stage_plan()

    @property
    def current_stage(self) -> DataStage:
        """Get current stage config."""
        return self.stages[self._current_stage_idx]

    @property
    def current_stage_idx(self) -> int:
        """Get current stage index."""
        return self._current_stage_idx

    def _validate_stages(self) -> None:
        """Validate stage configurations comprehensively."""
        training = self.job_config.training
        total_steps = training.steps

        # Check for duplicate stage names
        stage_names = [s.name for s in self.stages]
        if len(stage_names) != len(set(stage_names)):
            duplicates = [n for n in stage_names if stage_names.count(n) > 1]
            raise ValueError(f"Duplicate stage names found: {set(duplicates)}")

        # Validate each stage
        for i, stage in enumerate(self.stages):
            # Validate stage name
            if not stage.name or not stage.name.strip():
                raise ValueError(f"Stage at index {i} has empty or whitespace name")

            # Validate step ranges
            if stage.start_step < 0:
                raise ValueError(
                    f"Stage '{stage.name}' has invalid start_step: {stage.start_step}"
                )
            if stage.end_step is not None and stage.end_step <= stage.start_step:
                raise ValueError(
                    f"Stage '{stage.name}' has end_step ({stage.end_step}) <= "
                    f"start_step ({stage.start_step})"
                )

            # Validate required fields are set
            if stage.dataset_type is None:
                raise ValueError(
                    f"Stage '{stage.name}' must define 'dataset_type' "
                    "(e.g., 'huggingface', 'nanoset')"
                )
            if stage.seq_len is None:
                raise ValueError(f"Stage '{stage.name}' must define 'seq_len'")
            if stage.seq_len <= 0:
                raise ValueError(
                    f"Stage '{stage.name}' has invalid seq_len: {stage.seq_len}. "
                    "seq_len must be positive."
                )

            # Validate dataset_random_seed if provided
            if stage.dataset_random_seed is not None and stage.dataset_random_seed < 0:
                raise ValueError(
                    f"Stage '{stage.name}' has negative dataset_random_seed: "
                    f"{stage.dataset_random_seed}. Seeds must be non-negative."
                )

            # Validate start_step doesn't exceed total training steps
            if stage.start_step >= total_steps:
                raise ValueError(
                    f"Stage '{stage.name}' starts at step {stage.start_step} but "
                    f"training.steps is only {total_steps}. This stage would never run."
                )

            # Validate dataset source based on type
            if stage.dataset_type == "nanoset":
                if not stage.dataset_folders:
                    raise ValueError(
                        f"Stage '{stage.name}' with dataset_type='nanoset' "
                        "must define 'dataset_folders'"
                    )
                # Validate no empty or whitespace-only folder paths
                for j, folder in enumerate(stage.dataset_folders):
                    if not folder or not folder.strip():
                        raise ValueError(
                            f"Stage '{stage.name}' has empty or whitespace-only path "
                            f"in dataset_folders at index {j}"
                        )
            elif stage.dataset_type == "huggingface":
                if stage.dataset is None:
                    raise ValueError(
                        f"Stage '{stage.name}' with dataset_type='huggingface' "
                        "must define 'dataset'"
                    )
                # Validate dataset is not empty string
                if not stage.dataset.strip():
                    raise ValueError(
                        f"Stage '{stage.name}' has empty or whitespace-only 'dataset' value"
                    )

            # Validate dataset_weights if provided
            if stage.dataset_weights is not None:
                if not stage.dataset_weights:
                    raise ValueError(
                        f"Stage '{stage.name}' has empty dataset_weights list"
                    )
                # Check for NaN or inf values
                for j, w in enumerate(stage.dataset_weights):
                    if math.isnan(w):
                        raise ValueError(
                            f"Stage '{stage.name}' has NaN in dataset_weights "
                            f"at index {j}: {stage.dataset_weights}"
                        )
                    if math.isinf(w):
                        raise ValueError(
                            f"Stage '{stage.name}' has infinity in dataset_weights "
                            f"at index {j}: {stage.dataset_weights}"
                        )
                if any(w < 0 for w in stage.dataset_weights):
                    raise ValueError(
                        f"Stage '{stage.name}' has negative dataset_weights: "
                        f"{stage.dataset_weights}"
                    )
                if any(w > 1 for w in stage.dataset_weights):
                    raise ValueError(
                        f"Stage '{stage.name}' has dataset_weight > 1: "
                        f"{stage.dataset_weights}"
                    )
                # Check weights sum to 1 (with tolerance for floating point)
                weight_sum = sum(stage.dataset_weights)
                if abs(weight_sum - 1.0) > 0.001:
                    raise ValueError(
                        f"Stage '{stage.name}' dataset_weights must sum to 1.0, "
                        f"but sum is {weight_sum:.6f}: {stage.dataset_weights}"
                    )
                # Check weights match folders count for nanoset
                if stage.dataset_type == "nanoset" and stage.dataset_folders:
                    if len(stage.dataset_weights) != len(stage.dataset_folders):
                        raise ValueError(
                            f"Stage '{stage.name}' has {len(stage.dataset_weights)} "
                            f"weights but {len(stage.dataset_folders)} folders"
                        )

        # Check first stage starts at step 0
        if self.stages[0].start_step != 0:
            raise ValueError(
                f"First stage '{self.stages[0].name}' must start at step 0, "
                f"but starts at {self.stages[0].start_step}"
            )

        # Check for gaps or overlaps between stages
        for i in range(len(self.stages) - 1):
            current = self.stages[i]
            next_stage = self.stages[i + 1]

            # Determine current stage's end
            if current.end_step is not None:
                current_end = current.end_step
            else:
                # If no end_step, it should extend to next stage's start
                current_end = next_stage.start_step

            if current_end < next_stage.start_step:
                raise ValueError(
                    f"Gap in data stages: '{current.name}' ends at {current_end} "
                    f"but '{next_stage.name}' starts at {next_stage.start_step}. "
                    f"Steps {current_end} to {next_stage.start_step - 1} are not covered."
                )
            elif current_end > next_stage.start_step:
                raise ValueError(
                    f"Overlap in data stages: '{current.name}' ends at {current_end} "
                    f"but '{next_stage.name}' starts at {next_stage.start_step}. "
                    f"Steps {next_stage.start_step} to {current_end - 1} are covered "
                    f"by both stages."
                )

        # Check last stage covers until training end
        last_stage = self.stages[-1]
        if last_stage.end_step is not None:
            if last_stage.end_step < total_steps:
                raise ValueError(
                    f"Last stage '{last_stage.name}' ends at step {last_stage.end_step} "
                    f"but training.steps is {total_steps}. "
                    f"Steps {last_stage.end_step} to {total_steps - 1} are not covered. "
                    f"Remove 'end_step' from the last stage to cover until training end."
                )
            elif last_stage.end_step > total_steps:
                logger.warning(
                    f"Last stage '{last_stage.name}' end_step ({last_stage.end_step}) "
                    f"exceeds training.steps ({total_steps}). "
                    f"Training will end at step {total_steps}."
                )

    def _has_non_default_training_data(self, training) -> bool:
        """Check if [training] has non-default data fields set."""
        # Default values from Training dataclass
        defaults = {
            "dataset": "c4_test",
            "dataset_path": None,
            "dataset_type": "huggingface",
            "dataset_folders": [],
            "dataset_weights": None,
            # Note: dataset_random_seed and seq_len are not checked since they
            # have valid defaults that users commonly keep
        }
        return (
            training.dataset != defaults["dataset"]
            or training.dataset_path != defaults["dataset_path"]
            or training.dataset_type != defaults["dataset_type"]
            or training.dataset_folders != defaults["dataset_folders"]
            or training.dataset_weights != defaults["dataset_weights"]
        )

    def _format_training_data_fields(self, training) -> str:
        """Format [training] data fields for error message."""
        lines = []
        if training.dataset != "c4_test":
            lines.append(f"  dataset = {training.dataset!r}")
        if training.dataset_path is not None:
            lines.append(f"  dataset_path = {training.dataset_path!r}")
        if training.dataset_type != "huggingface":
            lines.append(f"  dataset_type = {training.dataset_type!r}")
        if training.dataset_folders:
            lines.append(f"  dataset_folders = {training.dataset_folders!r}")
        if training.dataset_weights is not None:
            lines.append(f"  dataset_weights = {training.dataset_weights!r}")
        return "\n".join(lines) if lines else "  (none detected)"

    def _log_stage_plan(self) -> None:
        """Log the data stage training plan."""
        logger.info("=" * 60)
        logger.info("DATA STAGE TRAINING PLAN")
        logger.info("=" * 60)
        logger.info(f"Total stages: {len(self.stages)}")
        logger.info("")

        training = self.job_config.training
        total_steps = training.steps

        for i, stage in enumerate(self.stages):
            effective = self.get_effective_config(stage)
            end_step = stage.end_step if stage.end_step is not None else total_steps
            stage_steps = end_step - stage.start_step

            # Calculate tokens for this stage
            # tokens = steps * global_batch_size * seq_len
            # Note: global_batch_size may be -1 (auto), so we show what we can
            global_bs = training.global_batch_size
            if global_bs > 0:
                tokens = stage_steps * global_bs * effective.seq_len
                token_str = f"{tokens / 1e9:.2f}B tokens"
            else:
                token_str = (
                    f"{stage_steps} steps × batch_size × {effective.seq_len} seq_len"
                )

            logger.info(f"Stage {i + 1}: {stage.name}")
            logger.info(
                f"  Steps: {stage.start_step:,} -> {end_step:,} ({stage_steps:,} steps)"
            )
            logger.info(f"  Estimated tokens: {token_str}")
            logger.info(f"  Dataset type: {effective.dataset_type}")

            if effective.dataset_folders:
                logger.info(
                    f"  Dataset folders: {len(effective.dataset_folders)} folders"
                )
                for folder in effective.dataset_folders[:3]:  # Show first 3
                    logger.info(f"    - {folder}")
                if len(effective.dataset_folders) > 3:
                    logger.info(
                        f"    ... and {len(effective.dataset_folders) - 3} more"
                    )
            else:
                logger.info(f"  Dataset: {effective.dataset}")

            if effective.dataset_weights:
                weights_str = ", ".join(
                    f"{w:.3f}" for w in effective.dataset_weights[:5]
                )
                if len(effective.dataset_weights) > 5:
                    weights_str += f", ... ({len(effective.dataset_weights)} total)"
                logger.info(f"  Weights: [{weights_str}]")

            logger.info(f"  Sequence length: {effective.seq_len}")
            logger.info("")

        logger.info("=" * 60)

    def get_effective_config(self, stage: DataStage) -> EffectiveStageConfig:
        """Get effective config for a stage. Each stage is self-contained."""
        # Use training.dataset_random_seed as fallback since it's optional
        training = self.job_config.training
        return EffectiveStageConfig(
            dataset=stage.dataset,
            dataset_path=stage.dataset_path,
            dataset_type=stage.dataset_type,
            dataset_folders=stage.dataset_folders,
            dataset_weights=stage.dataset_weights,
            dataset_random_seed=(
                stage.dataset_random_seed
                if stage.dataset_random_seed is not None
                else training.dataset_random_seed
            ),
            seq_len=stage.seq_len,
        )

    def find_stage_for_step(self, step: int) -> int:
        """Find the stage index for the given training step."""
        if step < 0:
            raise ValueError(f"Step cannot be negative, got {step}")
        for i, stage in enumerate(self.stages):
            in_range = step >= stage.start_step
            if stage.end_step is not None:
                in_range = in_range and step < stage.end_step
            elif i + 1 < len(self.stages):
                # If no end_step, use next stage's start_step
                in_range = in_range and step < self.stages[i + 1].start_step
            if in_range:
                return i
        # Default to last stage if step exceeds all ranges
        return len(self.stages) - 1

    def set_stage_for_step(self, step: int) -> bool:
        """Set current stage based on step. Returns True if stage changed."""
        new_idx = self.find_stage_for_step(step)
        if new_idx != self._current_stage_idx:
            old_stage = self.stages[self._current_stage_idx]
            self._current_stage_idx = new_idx
            new_stage = self.stages[new_idx]
            return True
        return False

    def maybe_transition_stage(self, step: int) -> bool:
        """Check if stage transition needed at this step. Returns True if transitioned."""
        new_idx = self.find_stage_for_step(step)
        if new_idx != self._current_stage_idx:
            old_stage = self.stages[self._current_stage_idx]
            new_stage = self.stages[new_idx]

            logger.info("=" * 60)
            logger.info("DATA STAGE TRANSITION")
            logger.info("=" * 60)
            logger.info(f"Step {step}: '{old_stage.name}' -> '{new_stage.name}'")

            old_effective = self.get_effective_config(old_stage)
            new_effective = self.get_effective_config(new_stage)

            # Log what changed
            changes = []
            if old_effective.dataset_weights != new_effective.dataset_weights:
                changes.append("dataset_weights")
            if old_effective.dataset_folders != new_effective.dataset_folders:
                changes.append("dataset_folders")
            if old_effective.seq_len != new_effective.seq_len:
                changes.append(
                    f"seq_len: {old_effective.seq_len} -> {new_effective.seq_len}"
                )

            if changes:
                logger.info(f"Changes: {', '.join(changes)}")
            else:
                logger.info("No config changes (stage name only)")

            if new_effective.dataset_weights:
                weights_str = ", ".join(
                    f"{w:.3f}" for w in new_effective.dataset_weights[:5]
                )
                if len(new_effective.dataset_weights) > 5:
                    weights_str += f", ... ({len(new_effective.dataset_weights)} total)"
                logger.info(f"New weights: [{weights_str}]")

            logger.info("=" * 60)

            self._current_stage_idx = new_idx
            return True
        return False

    def build_dataloader_for_stage(
        self, stage_idx: int | None = None
    ) -> BaseDataLoader:
        """Build dataloader for the specified stage (or current stage if None)."""
        if stage_idx is None:
            stage_idx = self._current_stage_idx

        stage = self.stages[stage_idx]
        effective = self.get_effective_config(stage)

        logger.info(f"Building dataloader for stage '{stage.name}' (idx={stage_idx})")

        # Temporarily override training config with stage-specific values
        training = self.job_config.training
        original_values = {}
        override_fields = [
            ("dataset", effective.dataset),
            ("dataset_path", effective.dataset_path),
            ("dataset_type", effective.dataset_type),
            ("dataset_folders", effective.dataset_folders),
            ("dataset_weights", effective.dataset_weights),
            ("dataset_random_seed", effective.dataset_random_seed),
            ("seq_len", effective.seq_len),
        ]

        for field_name, new_value in override_fields:
            original_values[field_name] = getattr(training, field_name)
            setattr(training, field_name, new_value)

        try:
            dataloader = self.build_dataloader_fn(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                job_config=self.job_config,
            )
        finally:
            # Restore original config values
            for field_name, original_value in original_values.items():
                setattr(training, field_name, original_value)

        return dataloader

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {"current_stage_idx": self._current_stage_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        if "current_stage_idx" in state_dict:
            old_idx = self._current_stage_idx
            self._current_stage_idx = state_dict["current_stage_idx"]
            if old_idx != self._current_stage_idx:
                logger.info(
                    f"Restored data stage index: {old_idx} -> {self._current_stage_idx} "
                    f"(stage: '{self.current_stage.name}')"
                )


class StageAwareDataloader(BaseDataLoader):
    """Dataloader wrapper that handles multi-stage training with proper checkpoint support.

    This wrapper:
    1. Manages the underlying dataloader for the current stage
    2. Saves/restores both stage index AND dataloader state for exact checkpoint resume
    3. Rebuilds dataloader on stage transitions

    The key insight for checkpoint correctness:
    - When saving: we save {stage_idx, dataloader_state_for_current_stage}
    - When loading: we restore stage_idx, rebuild dataloader for that stage,
      then restore the dataloader's internal state

    This ensures exact resume: same stage, same position within the dataset.
    """

    def __init__(
        self,
        stage_manager: DataStageManager,
        initial_dataloader: BaseDataLoader,
    ):
        self._stage_manager = stage_manager
        self._dataloader = initial_dataloader
        self._dp_rank = stage_manager.dp_rank
        self._dp_world_size = stage_manager.dp_world_size

    @property
    def dataloader(self) -> BaseDataLoader:
        """Get the underlying dataloader."""
        return self._dataloader

    def rebuild_for_current_stage(self) -> None:
        """Rebuild the underlying dataloader for the current stage."""
        self._dataloader = self._stage_manager.build_dataloader_for_stage()

    def maybe_transition(self, step: int) -> bool:
        """Check for stage transition and rebuild if needed. Returns True if transitioned."""
        if self._stage_manager.maybe_transition_stage(step):
            self.rebuild_for_current_stage()
            return True
        return False

    def __iter__(self) -> Iterator:
        """Iterate over the underlying dataloader."""
        return iter(self._dataloader)

    def __len__(self) -> int:
        """Return length of underlying dataloader if available."""
        return len(self._dataloader)

    def state_dict(self) -> dict[str, Any]:
        """Save state for checkpointing.

        Saves:
        - stage_idx: Which stage we're in
        - dataloader_state: Position within current stage's dataset
        - dp_world_size: For validation on resume
        """
        return {
            "stage_idx": self._stage_manager.current_stage_idx,
            "stage_name": self._stage_manager.current_stage.name,
            "dataloader_state": self._dataloader.state_dict(),
            "world_size": self._dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        This is the critical method for exact checkpoint resume:
        1. Restore stage_idx to stage manager
        2. Rebuild dataloader for the correct stage
        3. Restore dataloader's internal state (position in dataset)
        """
        if not state_dict:
            return

        # Validate world size consistency
        if "world_size" in state_dict:
            saved_world_size = state_dict["world_size"]
            if saved_world_size != self._dp_world_size:
                raise ValueError(
                    f"Data parallel world size changed from {saved_world_size} to "
                    f"{self._dp_world_size}. Dataloader state is incompatible."
                )

        # Restore stage index
        if "stage_idx" in state_dict:
            saved_stage_idx = state_dict["stage_idx"]
            saved_stage_name = state_dict.get("stage_name", "unknown")
            num_stages = len(self._stage_manager.stages)

            # Validate stage_idx is within bounds
            if saved_stage_idx < 0 or saved_stage_idx >= num_stages:
                raise ValueError(
                    f"Checkpoint stage_idx ({saved_stage_idx}) is out of bounds. "
                    f"Current config has {num_stages} stages (indices 0-{num_stages - 1}). "
                    f"Checkpoint was at stage '{saved_stage_name}'. "
                    "The stage configuration may have changed since the checkpoint was saved."
                )

            current_stage_idx = self._stage_manager.current_stage_idx

            if saved_stage_idx != current_stage_idx:
                logger.info(
                    f"Checkpoint was at stage '{saved_stage_name}' (idx={saved_stage_idx}), "
                    f"rebuilding dataloader..."
                )
                # Update stage manager's index
                self._stage_manager._current_stage_idx = saved_stage_idx
                # Rebuild dataloader for the restored stage
                self.rebuild_for_current_stage()

        # Restore dataloader state (position in dataset)
        if "dataloader_state" in state_dict:
            try:
                self._dataloader.load_state_dict(state_dict["dataloader_state"])
                logger.info("Restored dataloader position from checkpoint")
            except Exception as e:
                logger.warning(
                    f"Failed to restore dataloader state: {e}. "
                    "Training will resume from beginning of current stage's dataset."
                )


def build_stage_aware_dataloader(
    job_config: JobConfig,
    build_dataloader_fn: Callable,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any,
) -> tuple[StageAwareDataloader, DataStageManager]:
    """Build a stage-aware dataloader.

    Data stages are required. At least one stage must be defined in
    [[training.data_stages]].

    Returns:
        tuple of (StageAwareDataloader, DataStageManager)
    """
    stage_manager = DataStageManager(
        job_config=job_config,
        build_dataloader_fn=build_dataloader_fn,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
    )

    # Build initial dataloader for stage 0 (or whichever stage step 0 falls into)
    initial_dataloader = stage_manager.build_dataloader_for_stage()
    dataloader = StageAwareDataloader(stage_manager, initial_dataloader)
    logger.info(f"Created StageAwareDataloader with {len(stage_manager.stages)} stages")

    return dataloader, stage_manager
