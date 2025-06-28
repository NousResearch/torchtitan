# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


class PreprocessedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if dataset_path is not None:
            ds = load_from_disk(dataset_path)
        else:
            ds = load_dataset(dataset_name, split="train")

        logger.info(f"Loaded preprocessed dataset with {len(ds)} samples")

        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.dataset_name = dataset_name

        # Variables for checkpointing
        self._sample_idx = 0

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1

                keys = list(sample.keys())

                inputs = sample["inputs"]
                labels = sample["labels"] if "labels" in keys else inputs

                inputs = torch.LongTensor(inputs[:-1])
                labels = torch.LongTensor(labels[1:])

                args = {
                    "input": inputs,
                }
                if "position_ids" in keys:
                    args["position_ids"] = torch.LongTensor(sample["position_ids"][:-1])

                yield args, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def __len__(self):
        return len(self._data)

    def load_state_dict(self, state_dict):

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def collate_fn(batch, seq_len):
    # Unpack the batch
    inputs, labels = zip(*batch)

    # Stack tensors, padding to longest sampple
    args = {
        "input": torch.stack(
            [
                torch.cat(
                    [
                        x["input"],
                        torch.zeros(seq_len - len(x["input"]), dtype=torch.long),
                    ]
                )
                for x in inputs
            ]
        ),
    }
    if len(inputs) > 0:
        keys = list(inputs[0].keys())
        if "position_ids" in keys:
            args["position_ids"] = torch.stack(
                [
                    torch.cat(
                        [
                            x["position_ids"],
                            torch.arange(
                                seq_len - len(x["position_ids"]), dtype=torch.long
                            ),
                        ]
                    )
                    for x in inputs
                ]
            )

    labels = torch.stack(
        [
            torch.cat([x, torch.full((seq_len - len(x),), -100, dtype=torch.long)])
            for x in labels
        ]
    )

    return args, labels


def build_preprocessed_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    ds = PreprocessedDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, seq_len),
    )
