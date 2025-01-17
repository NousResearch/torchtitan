import hashlib
import os
import pickle
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from datatrove.pipeline.tokens.merger import load_doc_ends
from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.logging import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from numba import jit

def normalize(weights: List[float]) -> List[np.array]:
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        token_size (int): Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
        doc_offsets=False,
        use_cached_doc_offsets=True,
        return_loss_mask=False,
    ) -> None:

        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.token_size = token_size
        self.random_seed = random_seed
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=token_size,
                    shuffle=False,
                )
            )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()

        self.sequence_offsets = None
        if doc_offsets:
            cache_filename = hashlib.sha256(self.fingerprint().encode()).hexdigest()[:16] + ".packing"
            if use_cached_doc_offsets:
                if os.path.exists(cache_filename):
                    logger.info(f"Loading cached sequence offsets from {cache_filename}")
                    self.sequence_offsets = pickle.load(open(cache_filename, "rb"))

            if self.sequence_offsets is None:
                self.doc_ends_per_file = []
                for dataset in self.datatrove_datasets:
                    file_doc_ends = []
                    for file_dataset in dataset.files:
                        index_path = file_dataset.file_path + ".index"
                        with open(index_path, "rb") as f:
                            doc_ends = load_doc_ends(f)
                            file_doc_ends.append(doc_ends)
                    self.doc_ends_per_file.append(file_doc_ends)

                logger.info("Calculating sequence to file map")
                sequence_to_file_map = calculate_file_mapping(len(self), self.dataset_index, self.dataset_sample_index, [x.lens for x in self.datatrove_datasets])
                logger.info("Calculating sequence offsets")
                self.sequence_offsets = self._calculate_sequence_offsets(sequence_to_file_map)

                if use_cached_doc_offsets and ("LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0):
                    logger.info(f"Saving sequence offset cache to {cache_filename}")
                    pickle.dump(self.sequence_offsets, open(cache_filename, "wb"))

        self.loss_files = None
        if return_loss_mask:
            self.loss_files = []
            for dataset in self.datatrove_datasets:
                dataset_loss_files = []
                for file_dataset in dataset.files:
                    loss_path = file_dataset.file_path + ".loss"
                    if os.path.exists(loss_path):
                        dataset_loss_files.append(open(loss_path, "rb"))
                    else:
                        dataset_loss_files.append(None)
                self.loss_files.append(dataset_loss_files)

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        sample = self.datatrove_datasets[dataset][dataset_sample]

        if self.has_doc_offsets():
            sample["doc_offsets"] = torch.from_numpy(self.sequence_offsets[idx])

        if self.loss_files:
            dataset_lens = self.datatrove_datasets[dataset].lens
            file_idx = self.datatrove_datasets[dataset].current_file
            file_start = dataset_lens[file_idx] if file_idx >= 0 else 0
            file_offset = dataset_sample - file_start

            loss_file = self.loss_files[dataset][file_idx]
            offset = file_offset * (self.sequence_length + 1)
            loss_file.seek(offset)
            loss_values = np.frombuffer(loss_file.read(self.sequence_length + 1), dtype=np.bool_).copy()
            sample['loss_mask'] = torch.as_tensor(loss_values, dtype=torch.bool)

        return sample

    def build_nanoset_index(self) -> np.ndarray:
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        # num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1 
        num_epochs = 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        # dataset_index = dataset_index[: self.train_split_num_samples]
        # dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]

        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):

        logger.info(f"> Total number of samples: {len(self)}")
        logger.info(
            f"> Total number of tokens: {len(self) * self.sequence_length}")

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            logger.info(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                )

        if self.has_doc_offsets():
            logger.info("> Document offsets enabled")
        else:
            logger.info("> Document offsets disabled")

    def _calculate_sequence_offsets(self, sequence_to_file_map: Dict[int, Tuple[int, int, int]]) -> Dict[int, torch.Tensor]:
        offsets_map = []

        for idx in range(len(self)):
            dataset_idx, file_idx, file_offset = sequence_to_file_map[idx]
            doc_ends = self.doc_ends_per_file[dataset_idx][file_idx]

            start_pos = file_offset * (self.sequence_length + 1)
            end_pos = start_pos + self.sequence_length + 1

            doc_boundaries = np.searchsorted(doc_ends, [start_pos, end_pos])
            relevant_ends = doc_ends[doc_boundaries[0]:doc_boundaries[1]]

            sequence_relative_ends = relevant_ends - start_pos
            sequence_relative_ends = sequence_relative_ends[sequence_relative_ends > 0]

            if len(sequence_relative_ends) == 0:
                offsets_map.append(np.array([0, self.sequence_length + 1], dtype="int"))
            else:
                offsets_map.append(np.array([0] + sequence_relative_ends.tolist() + [self.sequence_length + 1], dtype="int"))

        return offsets_map

    def fingerprint(self) -> str:
        return ":".join(file.file_path for dataset in self.datatrove_datasets for file in dataset.files) \
            + "-" + "|".join([str(x) for x in self.dataset_weights]) \
            + "-" + str(self.sequence_length) \
            + "-" + str(self.token_size) \
            + "-" + str(self.random_seed)

    def has_doc_offsets(self) -> bool:
        return self.sequence_offsets is not None


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):

        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index

@jit(nopython=True, cache=True)
def bisect_right(a, x):
    lo=0
    hi=len(a)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo


def calculate_file_mapping(
    n_samples: int, dataset_index: np.ndarray, dataset_sample_index: np.ndarray, lenses: List[np.ndarray]
) -> Dict[int, Tuple[int, int, int]]:
    sequence_map = {}

    for idx in range(n_samples):
        dataset_idx = dataset_index[idx]
        sample_idx = dataset_sample_index[idx]

        lens = lenses[dataset_idx]
        # Use binary search to find which file contains this sample
        file_idx = bisect_right(lens, sample_idx) - 1

        # Calculate offset within the file
        file_start = lens[file_idx]
        file_offset = sample_idx - file_start

        sequence_map[idx] = (dataset_idx, file_idx, file_offset)

    return sequence_map


class IterableNanoset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        dataset_weights: Union[List[float], None] = None,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        random_seed: int = 1234,
        doc_offsets: bool = False,
        use_cached_doc_offsets: bool = True,
        loss_masking: bool = False,

    ) -> None:
        self.base_dataset = Nanoset(
            dataset_folders=dataset_folders,
            sequence_length=sequence_length,
            token_size=token_size,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
            doc_offsets=doc_offsets,
            use_cached_doc_offsets=use_cached_doc_offsets,
            return_loss_mask=loss_masking,
        )
        
        self.world_size = world_size
        self.rank = rank
        self.infinite = infinite
        
        self._sample_idx = 0
        
    def __iter__(self):
        while True:
            # calculate start and end indices for this rank
            start_idx = self._sample_idx + self.rank
            end_idx = len(self.base_dataset)
            
            for idx in range(start_idx, end_idx, self.world_size):
                sample = self.base_dataset[idx]
                tokens = sample["input_ids"]
                doc_offsets = sample.get("doc_offsets")
                loss_mask = sample.get("loss_mask")

                x = torch.LongTensor(tokens)
                input_ids = x[:-1]
                labels = x[1:]
                if loss_mask is not None:
                    labels = torch.where(loss_mask[1:], labels, torch.tensor(-100, dtype=labels.dtype))
                
                self._sample_idx = idx + self.world_size
                if doc_offsets is not None:
                    yield input_ids, labels, doc_offsets
                else:
                    yield input_ids, labels
            
            if not self.infinite:
                break
            else:
                self._sample_idx = 0
                logger.warning("Dataset is being re-looped")
    
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
    
    def state_dict(self):
        return {"sample_idx": self._sample_idx}

    # @staticmethod
    def collate_fn(batch):
        input_ids, labels, offsets = zip(*batch)
        max_offset_len = max(offset.size(0) for offset in offsets)

        padded_offsets = []
        for offset_tensor in offsets:
            padding_len = max_offset_len - offset_tensor.size(0)
            if padding_len > 0:
                # use the sequence length + 1 as the padding value (should be the last element already)
                pad_value = input_ids[0].size(0) + 1
                padding = torch.full((padding_len,), pad_value,
                                  dtype=offset_tensor.dtype,
                                  device=offset_tensor.device)
                padded_offsets.append(torch.cat([offset_tensor, padding]))
            else:
                padded_offsets.append(offset_tensor)

        return (
            torch.stack(input_ids),
            torch.stack(labels),
            torch.stack(padded_offsets)
        )


def build_nanoset_data_loader(
    dataset_folders: List[str],
    sequence_length: int,
    token_size: int,
    dataset_weights: Union[List[float], None],
    batch_size: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
    random_seed: int = 1234,
    doc_offsets: bool = False,
    use_cached_doc_offsets: bool = True,
    loss_masking: bool = False,
):
    """Build a data loader for Nanoset datasets."""
    nanoset = IterableNanoset(
        dataset_folders=dataset_folders,
        sequence_length=sequence_length,
        token_size=token_size,
        dataset_weights=dataset_weights,
        world_size=world_size,
        rank=rank,
        infinite=infinite,
        random_seed=random_seed,
        doc_offsets=doc_offsets,
        use_cached_doc_offsets=use_cached_doc_offsets,
        loss_masking=loss_masking,
    )
    return DPAwareDataLoader(rank, nanoset, batch_size=batch_size, collate_fn=IterableNanoset.collate_fn if doc_offsets else None)