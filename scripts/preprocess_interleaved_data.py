"""

Preprocessing script for interleaved data that contains both text-only and multimodal examples.
This script can process datasets that have examples like:

Text-only examples:
[
  {
    "text": "Some text content here..."
  }
]

Multimodal examples:
[
  {
    "messages": [
        {
            "role": "system",
            "content": [
              {"type": "text", "text": "You are a helpful assistant."}
              ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        },
        {
            "role": "assistant",
            "content": [
              {"type": "text", "text": "The image is a bee."}
            ]
        }
    ]
  }
]
"""
import argparse
import os
import shutil
import multiprocessing
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import random
import json
import base64
import uuid
from PIL import Image
import io
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from typing import List, Optional, Tuple
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset, Dataset as DatasetsDataset
from transformers import AutoTokenizer, AutoProcessor

from datetime import datetime, timedelta
import torch

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration


LOCAL_IMAGE_DIR = "./images"

SCHEMA = pa.schema(
    [
        pa.field("inputs", pa.large_list(pa.int32())),
        pa.field("labels", pa.large_list(pa.int32())),
        pa.field("position_ids", pa.large_list(pa.int32())),
        pa.field("sequence_lengths", pa.large_list(pa.int64())),
        pa.field("images", pa.large_list(pa.string())),
    ]
)

DATASET_INFO = r"""{
  "citation": "",
  "description": "",
  "features": {
    "inputs": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "labels": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "position_ids": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "sequence_lengths": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "images": {
      "feature": {
        "dtype": "str",
        "_type": "Value"
      },
      "_type": "LargeList"
    }
  },
  "homepage": "",
  "license": ""
}"""


def process_packing_shard(shard, args, tokenizer_pad_id, rank, world_size):
    packer = InterleavedPackedDataset(
        shard,
        max_seq_len=args.pack_to_sequence_length,
        padding_idx=tokenizer_pad_id,
        split_across_pack=not args.chat,
        show_pbar=rank == 0,
    )

    if args.save_to_disk:
        # create a schema that uses int64 for list sizes

        example = (
            {
                "inputs": packer.packs[0]["inputs"],
                "labels": packer.packs[0]["labels"],
                "position_ids": packer.packs[0]["position_ids"],
                "sequence_lengths": packer.packs[0]["sequence_lengths"],
                "images": packer.packs[0]["images"],
            }
            if len(packer.packs) > 0
            else None
        )

        oriented_data = {
            "inputs": [pack["inputs"] for pack in packer.packs],
            "labels": [pack["labels"] for pack in packer.packs],
            "position_ids": [pack["position_ids"] for pack in packer.packs],
            "sequence_lengths": [pack["sequence_lengths"] for pack in packer.packs],
            "images": [pack["images"] for pack in packer.packs],
        }
        pa_table = pa.Table.from_pydict(oriented_data, schema=SCHEMA)
        del oriented_data

        pa_ds.write_dataset(
            pa_table,
            os.path.join(args.save_to_disk, str(rank)),
            format="arrow",
        )

        filename = f"data-{rank:05d}-of-{world_size:05d}.arrow"

        shutil.move(
            os.path.join(args.save_to_disk, str(rank), "part-0.arrow"),
            os.path.join(args.save_to_disk, filename),
        )

        os.rmdir(os.path.join(args.save_to_disk, str(rank)))
    else:
        filename = None

    return packer.total_tokens, packer.packed_tokens, packer.dropped, filename, example


# https://github.com/pytorch/torchtune/blob/9d91fe39f08661952da4180b9e7fb2eba5a7a5e7/torchtune/datasets/_packed.py
class InterleavedPackedDataset(Dataset):
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
        group_size: int = 5000,
        show_pbar=True,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        self.packs = []
        self.previous_sample_boundary: int = 0
        self.packed_tokens: int = 0
        self.total_tokens: int = 0
        self.dropped: int = 0
        self.show_pbar = show_pbar
        self.group_size = group_size
        if split_across_pack:
            self._pack_greedy()
        else:
            self._pack_ffd()

    def _get_empty_pack(self):
        return {
            "inputs": np.empty(0, dtype=np.int32),
            "labels": np.empty(0, dtype=np.int32),
            "position_ids": np.empty(0, dtype=np.int32),
            "sequence_lengths": [],
            "images": [],
        }

    def _pack_ffd(self) -> None:
        ds_iterator = iter(self.ds)
        finished_iterating = False

        pbar = (
            tqdm(
                total=len(self.ds),
                desc="Packing dataset (FFD)",
                dynamic_ncols=True,
            )
            if self.show_pbar
            else None
        )

        while not finished_iterating:
            # 1. Fetch a large group of samples into memory.
            group = []
            try:
                for _ in range(self.group_size):
                    sample = next(ds_iterator)
                    seq_len = len(sample["inputs"])

                    if seq_len > self.max_seq_len:
                        self.dropped += 1
                        continue
                    # Store sample and its length for sorting
                    group.append({"sample": sample, "seq_len": seq_len})
            except StopIteration:
                finished_iterating = True

            if not group:
                break

            # 2. Sort the group by length in descending order (the "Decreasing" part of FFD).
            group.sort(key=lambda x: x["seq_len"], reverse=True)

            # 3. Pack this group using the "First-Fit" heuristic.
            # Each bin holds the samples it contains and its remaining space.
            bins = []  # List of {"samples": [], "remaining_space": max_seq_len}

            for item in group:
                placed = False
                # Try to place the item in the first available bin.
                for bin in bins:
                    if bin["remaining_space"] >= item["seq_len"]:
                        bin["samples"].append(item["sample"])
                        bin["remaining_space"] -= item["seq_len"]
                        placed = True
                        break

                # If no existing bin could accommodate the item, create a new one.
                if not placed:
                    bins.append(
                        {
                            "samples": [item["sample"]],
                            "remaining_space": self.max_seq_len - item["seq_len"],
                        }
                    )

            # 4. Convert the completed bins from this group into final, padded packs.
            for bin_info in bins:
                if self._should_stop_packing():
                    break

                current_pack = self._get_empty_pack()
                for sample in bin_info["samples"]:
                    tokens = np.array(sample["inputs"], dtype=np.int32)
                    labels = np.array(sample["labels"], dtype=np.int32)
                    images = sample.get("images", [])
                    seq_len = len(tokens)

                    current_pack["inputs"] = np.concatenate(
                        (current_pack["inputs"], tokens)
                    )
                    current_pack["labels"] = np.concatenate(
                        (current_pack["labels"], labels)
                    )
                    current_pack["position_ids"] = np.concatenate(
                        (
                            current_pack["position_ids"],
                            np.arange(seq_len, dtype=np.int32),
                        )
                    )
                    current_pack["sequence_lengths"].append(seq_len)
                    current_pack["images"].append(images)

                self._add_pack(current_pack)

            if pbar:
                pbar.update(len(group))

            if self._should_stop_packing():
                # Ensure the outer loop breaks if max_packs is reached.
                break

        if pbar:
            # Manually set pbar to total to show 100% at the end
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

    def _pack_greedy(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""

        current_pack = self._get_empty_pack()

        pbar = (
            tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)
            if self.show_pbar
            else None
        )

        for sample in self.ds:
            tokens = np.array(sample["inputs"], dtype=np.int32)
            labels = np.array(sample["labels"], dtype=np.int32)
            images = sample.get("images", [])

            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                # print(
                #     f"Dropping sample that is too long ({seq_len} > {self.max_seq_len})"
                # )
                self.dropped += 1
                continue

            current_pack["inputs"] = np.concatenate((current_pack["inputs"], tokens))
            current_pack["labels"] = np.concatenate((current_pack["labels"], labels))

            position_ids = np.arange(seq_len, dtype=np.int32)
            current_pack["position_ids"] = np.concatenate(
                (current_pack["position_ids"], position_ids)
            )

            current_pack["sequence_lengths"] += [seq_len]
            current_pack["images"].append(images)

            while (
                len(current_pack["inputs"]) > self.max_seq_len
                and not self._should_stop_packing()
            ):
                current_pack = self._split_and_add_pack(current_pack)

            if pbar:
                pbar.update()

            self.previous_sample_boundary = len(current_pack["inputs"])

            if self._should_stop_packing():
                break

        if len(current_pack["inputs"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
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
            "images": current_pack["images"][:-1],
        }

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
            "images": [current_pack["images"][-1]] if current_pack["images"] else [],
        }

    def _add_pack(self, pack) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _pad_pack(self, pack, padding_idx: int):
        """Pads a pack to ``self.max_seq_len``."""
        num_tokens = len(pack["inputs"])
        num_padding_tokens = self.max_seq_len - num_tokens

        self.packed_tokens += num_tokens
        self.total_tokens += self.max_seq_len

        padded_inputs = np.pad(
            pack["inputs"], (0, num_padding_tokens), constant_values=self.padding_idx
        )
        padded_labels = np.pad(
            pack["labels"], (0, num_padding_tokens), constant_values=-100
        )

        if num_padding_tokens > 0:
            # don't care much about padded position_ids, but create them for consistency
            start_pos = int(pack["position_ids"][-1] + 1) if num_tokens > 0 else 0
            pad_positions = np.arange(
                start_pos, start_pos + num_padding_tokens, dtype=np.int32
            )
            padded_position_ids = np.concatenate((pack["position_ids"], pad_positions))
        else:
            padded_position_ids = pack["position_ids"]

        padded_seq_lens = pack["sequence_lengths"]
        if num_padding_tokens > 0:
            padded_seq_lens.append(num_padding_tokens)

        return {
            "inputs": padded_inputs,
            "labels": padded_labels,
            "position_ids": padded_position_ids,
            "sequence_lengths": padded_seq_lens,
            "images": pack["images"],
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.packs[idx]


def main(args):
    from datasets import load_dataset

    if 'json' in args.dataset:
        dataset = load_dataset('json', data_files=args.dataset)['train']
        if args.limit is not None:
            dataset = dataset.select(range(args.limit))
    else:
        dataset = load_dataset(args.dataset, name=args.subset, split=args.split)

    def remove_none_recursively(obj):
        if isinstance(obj, dict):
            return {k: remove_none_recursively(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [remove_none_recursively(item) for item in obj]
        else:
            return obj

    # Initialize both tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    processor = AutoProcessor.from_pretrained(args.preprocessor, use_fast=True)

    def _detect_sample_type(sample):
        """Detect whether a sample is text-only or multimodal based on its structure."""
        if "text" in sample:
            return "text_only"
        elif "messages" in sample:
            return "multimodal"
        else:
            # Default to text-only if we can't determine
            return "text_only"

    def _tokenize_text_only(sample):
        """Tokenize text-only samples."""
        inputs = tokenizer.batch_encode_plus(sample["text"]).input_ids
        for x in inputs:
            x.append(tokenizer.eos_token_id)
        return {"inputs": inputs, "labels": inputs, "images": []}

    def _tokenize_chat_multimodal(sample):
        """Tokenize multimodal chat samples."""
        inputs = []
        labels = []
        images = []

        for conversation in sample["messages"]:
            image = None
            conversation = remove_none_recursively(conversation)

            for message in conversation:
                keys = list(message.keys())

                for item in message['content']:
                    if 'base64' in item.keys():
                        # save image in local folder as PIL image with uuid 
                        # Decode base64 image data
                        image_data = base64.b64decode(item['base64'])
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Generate UUID4 filename
                        image_filename = f"{uuid.uuid4()}.jpg"
                        image_path = os.path.join(LOCAL_IMAGE_DIR, image_filename)
                        
                        # Ensure directory exists
                        os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
                        
                        # Save image as JPG
                        image.save(image_path, 'JPEG')
                        
                        # remove base64 key, set type image
                        item.pop('base64')
                        item['type'] = 'image'
                        item['path'] = image_path

                        image = image_path

                if "from" in keys and "value" in keys:
                    # sharegpt format
                    message_from = message.pop("from")
                    if message_from == "gpt":
                        message["role"] = "assistant"
                    elif message_from == "human":
                        message["role"] = "user"
                    else:
                        message["role"] = message_from

                    message["content"] = message.pop("value")
                elif "role" in keys and "content" in keys:
                    pass
                else:
                    raise RuntimeError(f"Unknown chat format, keys are {keys}")

            tokenized = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")

            tokens = tokenized["input_ids"][0]

            current_len = 0
            label = []
            for i in range(len(conversation)):
                if i + 1 == len(conversation):
                    next_tokens = processor.apply_chat_template(conversation, 
                    tokenize=True, return_dict=True, return_tensors="pt")["input_ids"][0][current_len:]
                else:
                    if "assistant" == conversation[i + 1]["role"]:
                        next_tokens = processor.apply_chat_template(conversation[: i + 1], 
                        add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"][0][current_len:]
                    else:
                        next_tokens = processor.apply_chat_template(conversation[: i + 1], 
                        tokenize=True, return_dict=True)["input_ids"][0][current_len:]

                if conversation[i]["role"] == "assistant":
                    label.extend(next_tokens)
                else:
                    label.extend([-100] * len(next_tokens))

                current_len += len(next_tokens)

            inputs.append(tokens)
            labels.append(label)
            images.append(image)

        return {
            "inputs": inputs,
            "labels": labels,
            "images": images,
        }

    def _tokenize_chat_text_only(sample):
        """Tokenize text-only chat samples."""
        inputs = []
        labels = []

        for conversation in sample["conversations"]:
            for message in conversation:
                keys = list(message.keys())

                if "from" in keys and "value" in keys:
                    # sharegpt format
                    message_from = message.pop("from")
                    if message_from == "gpt":
                        message["role"] = "assistant"
                    elif message_from == "human":
                        message["role"] = "user"
                    else:
                        message["role"] = message_from

                    message["content"] = message.pop("value")
                elif "role" in keys and "content" in keys:
                    pass
                else:
                    raise RuntimeError(f"Unknown chat format, keys are {keys}")

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
            "images": [],
        }

    def _tokenize_interleaved(sample):
        """Main tokenization function that handles both text-only and multimodal samples."""
        sample_type = _detect_sample_type(sample)
        
        if sample_type == "text_only":
            if "text" in sample:
                return _tokenize_text_only(sample)
            elif "conversations" in sample:
                return _tokenize_chat_text_only(sample)
            else:
                raise ValueError(f"Unknown text-only format: {list(sample.keys())}")
        elif sample_type == "multimodal":
            return _tokenize_chat_multimodal(sample)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    dataset = dataset.shuffle(args.seed)

    original_column_names = list(dataset.features.keys())

    dataset = dataset.map(
        _tokenize_interleaved,
        batched=True,
        #batch_size=args.batch_size,
    )

    dataset = dataset.remove_columns(original_column_names)

    efficiency = 1.0
    dropped = 0
    if args.pack_to_sequence_length:
        num_shards = 32  # args.num_proc
        shards = [
            dataset.shard(num_shards=num_shards, index=i) for i in range(num_shards)
        ]

        with multiprocessing.Pool(processes=num_shards) as pool:
            process_args = [
                (shard, args, tokenizer.pad_token_id, index, num_shards)
                for index, shard in enumerate(shards)
            ]

            results = pool.starmap(process_packing_shard, process_args)

        examples = []
        filenames = []
        total_tokens = 0
        packed_tokens = 0

        for total, packed, dropped_, filename, example in tqdm(results):
            if example:
                examples.append(example)
            if filename:
                filenames.append(filename)
            total_tokens += total
            packed_tokens += packed
            dropped += dropped_

        if total_tokens > 0:
            efficiency = packed_tokens / total_tokens

        example = examples[0]

        if args.save_to_disk:
            with open(os.path.join(args.save_to_disk, "dataset_info.json"), "wb") as f:
                f.write(DATASET_INFO.encode())

            # verify we can open and do any conversion needed
            dataset = load_dataset(args.save_to_disk, num_proc=args.num_proc)

    else:
        if args.drop_larger_than:
            len_before = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x["inputs"]) <= args.drop_larger_than
            )
            dropped = len_before - len(dataset)

        if args.save_to_disk:
            print(f"Saving to {args.save_to_disk}")
            dataset.save_to_disk(args.save_to_disk)

        example = dataset[0]

    if args.show_example:
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

    if dropped > 0:
        print(f"Dropped {dropped} too-long samples")
    print(f"Efficiency: {efficiency * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--preprocessor", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--multiturn-only", action="store_true")
    parser.add_argument("--pack-to-sequence-length", type=int)
    parser.add_argument("--drop-larger-than", type=int)
    parser.add_argument("--save-to-disk", type=str)
    parser.add_argument("--show-example", action="store_true")
    args = parser.parse_args()

    main(args)
