import torchtitan
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

from .hf_datasets import build_hf_dataloader
from .nanoset import build_nanoset_dataloader

def build_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
):
    """Build the appropriate dataloader function"""

    if job_config.training.dataset_type == "huggingface":
        return build_hf_dataloader(dp_world_size, dp_rank, tokenizer, job_config, infinite)
    elif job_config.training.dataset_type == "nanoset":
        return build_nanoset_dataloader(dp_world_size, dp_rank, tokenizer, job_config, infinite)
    else:
        raise ValueError(f"Unknown dataset type `{job_config.training.dataset_type}`")
