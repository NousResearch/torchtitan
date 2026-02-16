# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

from datasets import Dataset, load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config import JobConfig
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
        shuffle_seed: int | None = 42,
    ) -> None:
        dataset_id = dataset_path if dataset_path is not None else dataset_name

        try:
            logger.info(
                f"Loading preprocessed dataset {dataset_id} for rank {dp_rank} via load_dataset"
            )
            ds = load_dataset(dataset_id, split="train")
        except Exception:
            logger.info(
                f"Loading preprocessed dataset {dataset_id} for rank {dp_rank} via load_from_disk"
            )
            ds = load_from_disk(dataset_id)

        if not isinstance(ds, Dataset):
            raise ValueError(
                f"Expected the loaded dataset to be a `Dataset`, but got {type(ds)}"
            )

        if shuffle_seed is not None:
            ds = ds.shuffle(shuffle_seed)

        logger.info(f"Loaded preprocessed dataset with {len(ds)} samples")

        self._data: Dataset = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.dataset_name = dataset_name
        self.dp_rank = dp_rank

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
            sample: dict[str, Any]
            for sample in self._get_data_iter():
                self._sample_idx += 1

                inputs = torch.tensor(sample["inputs"], dtype=torch.int64)
                labels = torch.tensor(
                    sample["labels"] if "labels" in sample else sample["inputs"],
                    dtype=torch.int64,
                )

                labels = torch.roll(labels, shifts=-1, dims=0)
                labels[-1] = -100

                args = {
                    "input": inputs,
                }
                if "position_ids" in sample:
                    args["position_ids"] = torch.tensor(
                        sample["position_ids"], dtype=torch.int64
                    )
                if "sequence_lengths" in sample:
                    args["sequence_lengths"] = torch.tensor(
                        sample["sequence_lengths"], dtype=torch.int64
                    )

                yield args, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
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


def collate_fn(batch):
    inputs, labels = zip(*batch)

    expected_len = len(inputs[0]["input"])
    for i, (input_item, label_item) in enumerate(batch):
        input_len = len(input_item["input"])
        label_len = len(label_item)

        if input_len != expected_len or label_len != expected_len:
            raise ValueError(
                f"All tensors in the batch must have the same length. "
                f"Expected length {expected_len} (from item 0), but item {i} has "
                f"an input length of {input_len} and a label length of {label_len}."
            )

    args = {"input": torch.stack([x["input"] for x in inputs])}

    if "position_ids" in inputs[0]:
        args["position_ids"] = torch.stack([x["position_ids"] for x in inputs])
    if "sequence_lengths" in inputs[0]:
        args["sequence_lengths"] = torch.stack([x["sequence_lengths"] for x in inputs])

    labels_tensor = torch.stack(list(labels))

    return args, labels_tensor


def build_preprocessed_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for pre-processed datasets."""
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
        collate_fn=collate_fn,
    )
