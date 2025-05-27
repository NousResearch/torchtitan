import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
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
                    shuffle=True,
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

        return self.datatrove_datasets[dataset][dataset_sample]

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
    ) -> None:
        self.base_dataset = Nanoset(
            dataset_folders=dataset_folders,
            sequence_length=sequence_length,
            token_size=token_size,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
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
                
                x = torch.LongTensor(tokens)
                input_ids = x[:-1]
                labels = x[1:]
                
                self._sample_idx = idx + self.world_size
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
    )
    return DPAwareDataLoader(rank, nanoset, batch_size=batch_size, world_size=world_size)