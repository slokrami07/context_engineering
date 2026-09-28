"""Tiktoken-backed Tokenizer implementation."""

from collections.abc import Sequence

from context_engineer.errors import TokenizerUnavailable
from context_engineer.tokenizers.base import Tokenizer
from context_engineer.types import Message


class TiktokenTokenizer(Tokenizer):
    """Accurate, deterministic tokenizer powered by tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise TokenizerUnavailable(
                f"tiktoken library is required for TiktokenTokenizer ({e}). "
                "Install with: pip install 'context-engineer[tiktoken]'"
            ) from e

        self.id = f"tiktoken:{encoding_name}"
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        # Disallowing special tokens ensures control tokens in user content (<|im_end|>) are treated as literal text (K7)
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self._encoding.decode(ids)

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        total = 0
        for m in messages:
            # ChatML framing: <|im_start|>{role}\n{content}<|im_end|>\n
            role_tokens = len(self.encode(m.role))
            content_tokens = len(self.encode(m.content))
            total += role_tokens + content_tokens + 4
        if add_generation_prompt:
            # Priming: <|im_start|>assistant\n = 3 tokens
            total += 3
        return total
