from typing import List, Tuple
from enum import Enum, auto

from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer


class ChatFormat(Enum):
    LLAMA3 = auto()
    CHATML = auto()


class ChatTokenizer:
    def __init__(
        self, tokenizer: TikTokenizer, chat_format: ChatFormat = ChatFormat.LLAMA3,
    ):
        self.tokenizer = tokenizer

        self._chat_format = chat_format
        if chat_format == ChatFormat.LLAMA3:
            self._header_start = "<|start_header_id|>"
            self._header_end = "<|end_header_id|>\n\n"
            self._turn_end = "<|eot_id|>"
        elif chat_format == ChatFormat.CHATML:
            self._header_start = "<|im_start|>"
            self._header_end = "\n"
            self._turn_end = "<|im_end|>\n"

    def __call__(self, conversation: List[dict]) -> Tuple[List[int], List[bool]]:
        tokens = []
        # Append <|begin_of_text|>
        tokens.append(self.tokenizer.bos_id)
        is_completitions = [False] * len(tokens)

        for message in conversation:
            message_tokens, message_completitions = self.encode_message(message)
            tokens.extend(message_tokens)
            is_completitions.extend(message_completitions)

        tokens.append(self.tokenizer.eos_id)
        is_completitions.append(False)

        return tokens, is_completitions

    def encode_message(self, message: dict) -> Tuple[List[int], List[int]]:
        role, is_input = self._get_role(message)

        # Encode header
        tokens = self.tokenizer.encode(
            f"{self._header_start}{role}{self._header_end}", bos=False, eos=False
        )
        is_completitions = [False] * len(tokens)

        # Encode message
        tokens.extend(
            self.tokenizer.encode(message["value"].strip(), bos=False, eos=False)
        )

        # Append <|eot_id|> token
        tokens.extend(self.tokenizer.encode(self._turn_end, bos=False, eos=False))

        # True if token belongs to assistant answer, False otherwise
        is_completitions.extend([not is_input] * (len(tokens) - len(is_completitions)))

        return tokens, is_completitions

    def _get_role(self, message: dict) -> Tuple[str, bool]:
        """
        Return the canonical role for a given message, as well as if its value
        should be considered input (and therefore not trained on)
        """
        role = message["from"]
        if role == "gpt":
            role = "assistant"
        return role, role == "user"

