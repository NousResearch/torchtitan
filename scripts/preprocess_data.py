import argparse
import torch

from typing import Optional
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset, Dataset as DatasetsDataset
from transformers import AutoTokenizer


# https://github.com/pytorch/torchtune/blob/9d91fe39f08661952da4180b9e7fb2eba5a7a5e7/torchtune/datasets/_packed.py
class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples with a ``Sampler`` as part of the dataloader. Currently,
    this only supports in-memory map-style datasets.

    The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

    The general flow on initialization is: load tokenized sample -> add to buffer ->
    when buffer is long enough, add to ``self.packs``.

    During training, returns self.packs[idx] as input, label, attention mask, and
    position ids. The attention mask is a lower triangular block mask to prevent
    samples from cross-attending within a pack. The position ids indicate the position
    of each token relative to its sample within a pack. These are all padded to max
    sequence length, so a batch-wise collator is not needed.

    A packed sample is made up of individual smaller sequence length samples jammed together
    within ``max_seq_len``. For example, if max_seq_len is 6 and there are varied
    length samples::

        tokens = [
            [S1, S1, S1, S2, S2, pad],
            [S3, S3, S4, S4, pad, pad],
            ...,
        ]

    To prevent cross-contamination, the following mask would be returned for the
    first pack in the example::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    The position ids would be::

        input_pos = [
            [0, 1, 2, 0, 1, 2],
            [0, 1, 0, 1, 2, 3],
            ...,
        ]

    The identity matrix is used in the mask for pad tokens instead of a causal mask.
    For position ids for pad tokens, we simply continue to increment from the previous
    sample normally.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
            packs as possible.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs = []
        self.previous_sample_boundary: int = 0
        self._pack()

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "inputs": [],
            "labels": [],
            "position_ids": [],
            "sequence_lengths": [],
        }

        # Only show progress bar on rank 0
        pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

        for sample in self.ds:
            tokens, labels = sample["inputs"], sample["labels"]

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                # print(
                #     f"Dropping sample that is too long ({seq_len} > {self.max_seq_len})"
                # )
                continue

            # Update the current pack
            current_pack["inputs"] += tokens
            current_pack["labels"] += labels
            current_pack["position_ids"] += [
                x % self.max_seq_len for x in range(seq_len)
            ]
            current_pack["sequence_lengths"] += [seq_len]

            # If the current pack is over the max_seq_len, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            while (
                len(current_pack["inputs"]) > self.max_seq_len
                and not self._should_stop_packing()
            ):
                current_pack = self._split_and_add_pack(current_pack)

            pbar.update()

            # Keep track of previous sample boundary
            self.previous_sample_boundary = len(current_pack["inputs"])

            if self._should_stop_packing():
                break

        # Handle the last pack if there's leftover and we haven't filled up the max packs
        if len(current_pack["inputs"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            # No need to handle splitting at this point so we can just add the current pack
            self._add_pack(current_pack)

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack):
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.max_seq_len
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "inputs": current_pack["inputs"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "position_ids": current_pack["position_ids"][:boundary],
            "sequence_lengths": current_pack["sequence_lengths"][:-1] + seq_len_padding,
        }

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["inputs"][boundary:])
            if self.split_across_pack
            else current_pack["sequence_lengths"][-1]
        )

        return {
            "inputs": current_pack["inputs"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "position_ids": current_pack["position_ids"][boundary:],
            "sequence_lengths": [next_seq_len],
        }

    def _add_pack(self, pack) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _pad_pack(self, pack, padding_idx: int):
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        num_padding_tokens = self.max_seq_len - len(pack["inputs"])
        padded_tokens = pack["inputs"] + [padding_idx] * num_padding_tokens

        # Pad labels
        padded_labels = pack["labels"] + [-100] * num_padding_tokens

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        padded_seq_lens = pack["sequence_lengths"]
        if num_padding_tokens > 0:
            padded_seq_lens += [num_padding_tokens]

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        start_pos = pack["position_ids"][-1] + 1 if pack["position_ids"] else 0
        num_positions_to_generate = self.max_seq_len - len(pack["position_ids"])
        new_positions = range(start_pos, start_pos + num_positions_to_generate)
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_new_positions = [min(p, self.max_seq_len - 1) for p in new_positions]
        padded_input_pos = pack["position_ids"] + clamped_new_positions

        return {
            "inputs": padded_tokens,
            "labels": padded_labels,
            "position_ids": padded_input_pos,
            "sequence_lengths": padded_seq_lens,
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.packs[idx]


def main(args):
    dataset = load_dataset(args.dataset, name=args.subset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def _tokenize(sample):
        # assumes "text" is the column
        inputs = tokenizer.batch_encode_plus(sample["text"]).input_ids
        for x in inputs:
            x.append(tokenizer.eos_token_id)
        return {"inputs": inputs}

    def _tokenize_chat(sample):
        inputs = []
        labels = []

        for conversation in sample["conversations"]:
            for message in conversation:

                message_from = message.pop("from")
                if message_from == "gpt":
                    message["role"] = "assistant"
                elif message_from == "human":
                    message["role"] = "user"
                else:
                    message["role"] = message_from

                message["content"] = message.pop("value")

            tokens = tokenizer.apply_chat_template(conversation, tokenize=True)
            label = []

            current_len = 0
            for i in range(len(conversation)):
                if i + 1 == len(conversation):
                    next_tokens = tokenizer.apply_chat_template(conversation)[
                        current_len:
                    ]
                else:
                    if "assistant" == conversation[i + 1]["role"]:
                        next_tokens = tokenizer.apply_chat_template(
                            conversation[: i + 1], add_generation_prompt=True
                        )[current_len:]
                    else:
                        next_tokens = tokenizer.apply_chat_template(
                            conversation[: i + 1]
                        )[current_len:]

                if conversation[i]["role"] == "assistant":
                    label.extend(next_tokens)
                else:
                    label.extend([-100] * len(next_tokens))

                current_len += len(next_tokens)

            inputs.append(tokens)
            labels.append(label)

        return {
            "inputs": inputs,
            "labels": labels,
        }

    dataset = dataset.shuffle(args.seed)
    if args.limit:
        dataset = dataset.select(range(args.limit))
    if args.chat and args.multiturn_only:
        dataset = dataset.filter(lambda x: len(x["conversations"]) > 3)

    original_column_names = list(dataset.features.keys())
    dataset = dataset.map(
        _tokenize_chat if args.chat else _tokenize,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )
    dataset = dataset.remove_columns(original_column_names)

    if args.pack_to_sequence_length:
        dataset = PackedDataset(
            dataset,
            max_seq_len=args.pack_to_sequence_length + 1, # one extra, so that after causal shift we're at the sequence length
            padding_idx=tokenizer.pad_token_id,
            max_packs=args.limit,
            split_across_pack=not args.chat,
        )
        column_names = ["inputs", "labels", "position_ids", "sequence_lengths"]
        oriented_dataset = {key: [] for key in column_names}
        for row in dataset:
            for key in column_names:
                oriented_dataset[key].append(row[key])
        dataset = DatasetsDataset.from_dict(oriented_dataset)

    example = dataset[0]
    inputs = example["inputs"]
    labels = example["labels"] if "labels" in example else None
    position_ids = example["position_ids"] if "position_ids" in example else None

    example_out = ""
    for i in range(0, len(inputs)):
        token = inputs[i]
        label = labels[i] if labels is not None else token
        position_id = position_ids[i] if position_ids is not None else None

        decoded = tokenizer.decode(token)

        if label == -100:
            example_out += f"\033[31m{decoded}\033[0m({token}"
        else:
            example_out += f"\033[32m{decoded}\033[0m({token}"
        
        if position_id != None:
            example_out += f"@{position_id})"
        else:
            example_out += ")"

    print(example_out)

    if args.save_to_disk:
        dataset.save_to_disk(args.save_to_disk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--multiturn-only", action="store_true")
    parser.add_argument("--pack-to-sequence-length", type=int)
    parser.add_argument("--save-to-disk", type=str)
    args = parser.parse_args()

    main(args)
