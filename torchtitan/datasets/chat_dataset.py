import torch


from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.tokenizer.chat_tokenizer import ChatTokenizer
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node


class ChatDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: TikTokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        converstations_column: str = "conversations",
        split: str = "train",
        pad_to: int = 1,
    ) -> None:
        ds = load_dataset(dataset_name)[split]

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = ChatTokenizer(tokenizer)
        self.seq_len = seq_len
        self.infinite = infinite
        self._converstations_column = converstations_column
        self._pad_to = pad_to

        # Variables for checkpointing
        self._sample_idx = 0

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):

        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                input_ids, is_completions = self._tokenizer(
                    sample[self._converstations_column]
                )

                input_ids = input_ids[: self.seq_len + 1]
                is_completions = is_completions[: self.seq_len + 1]

                labels = [x if y else -100 for x, y in zip(input_ids, is_completions)]

                input_ids = input_ids[:-1]
                labels = labels[1:]

                rem = len(input_ids) % self._pad_to
                if rem > 0:
                    input_ids.extend(
                        [self._tokenizer.tokenizer.eos_id] * (self._pad_to - rem)
                    )
                    labels.extend([-100] * (self._pad_to - rem))

                yield torch.LongTensor(input_ids), torch.LongTensor(labels)

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


def build_chat_data_loader(
    dataset_name: str,
    tokenizer: TikTokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    pad_to: int = 1,
):
    hf_ds = ChatDataset(
        dataset_name, tokenizer, seq_len, world_size, rank, infinite, pad_to=pad_to
    )
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
