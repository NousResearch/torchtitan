from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from transformers import AutoTokenizer

class AutoHfTokenizer(Tokenizer):
    def __init__(self, tokenizer_path: str):
        self.auto_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(self, *args, **kwargs) -> list[int]:
        return self.auto_tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        return self.auto_tokenizer.decode(*args, **kwargs)

    @property
    def n_words(self) -> int:
        return len(self.auto_tokenizer)
    
    @property
    def eos_id(self) -> int:
        self.auto_tokenizer.eos_token_id


def build_auto_tokenizer(job_config: JobConfig) -> AutoHfTokenizer:
    return AutoHfTokenizer(job_config.model.tokenizer_path)
