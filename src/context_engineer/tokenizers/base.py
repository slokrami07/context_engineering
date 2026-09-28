"""Tokenizer protocol defining token counting, encoding, decoding, and message template counting."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from context_engineer.types import Message


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers used across all budget calculations and render gates."""

    id: str

    def count(self, text: str) -> int:
        """Counts tokens for raw text string."""
        ...

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Counts total tokens for rendered messages including chat template overhead."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encodes text to a sequence of integer token IDs."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decodes integer token IDs back to a string."""
        ...
