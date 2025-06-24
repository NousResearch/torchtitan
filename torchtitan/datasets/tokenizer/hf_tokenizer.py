from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from transformers import AutoTokenizer

class HfTokenizer(Tokenizer):
    def __init__(self, tokenizer_path: str):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, *args, **kwargs) -> list[int]:
        return self.hf_tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        return self.hf_tokenizer.decode(*args, **kwargs)

    @property
    def n_words(self) -> int:
        return len(self.hf_tokenizer)
    
    @property
    def eos_id(self) -> int:
        self.hf_tokenizer.eos_token_id


def build_hf_tokenizer(job_config: JobConfig) -> HfTokenizer:
    return HfTokenizer(job_config.model.tokenizer_path)
