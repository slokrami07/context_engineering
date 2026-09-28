"""Approximate zero-dependency, offline-guaranteed Tokenizer."""

import math
from collections.abc import Sequence

from context_engineer.tokenizers.base import Tokenizer
from context_engineer.types import Message


class ApproxTokenizer(Tokenizer):
    """Fast, deterministic, zero-dependency tokenizer with guaranteed len(encode(t)) == count(t)."""

    def __init__(self, id: str = "approx", chars_per_token: float = 3.5) -> None:
        self.id = id
        self._chars_per_token = max(1.0, chars_per_token)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._chars_per_token))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        token_count = self.count(text)
        # Create deterministic integer IDs chunked across the text to guarantee len(encode(t)) == count(t)
        # Each ID encodes the ord value of the leading character in each chunk
        step = max(1, len(text) / token_count)
        return [ord(text[int(i * step)]) for i in range(token_count)]

    def decode(self, ids: list[int]) -> str:
        # Reconstruct characters from integer IDs
        return "".join(chr(i) if 0 <= i <= 0x10FFFF else "?" for i in ids)

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        total = 0
        for m in messages:
            total += self.count(m.role) + self.count(m.content) + 4
        if add_generation_prompt:
            total += 3
        return total
