"""Protocols defining standard interfaces for tokenizers, retrievers, summarizers, and redactors."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from context_engineer.tokenizers.base import Tokenizer
from context_engineer.types import Turn


@runtime_checkable
class Retriever(Protocol):
    """Protocol for memory/context retrieval across evicted turns."""

    def retrieve(
        self,
        query: str,
        candidates: Sequence[Turn],
        *,
        k: int,
        token_budget: int,
        tok: Tokenizer,
    ) -> list[Turn]:
        """Retrieves top-k relevant turns bounded within the allocated token budget."""
        ...


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for abstracting dropped turns into narrative context."""

    id: str
    version: str

    def summarize(
        self,
        dropped: Sequence[Turn],
        *,
        max_tokens: int,
        tok: Tokenizer,
    ) -> str:
        """Generates an abstract narrative summary for evicted turns."""
        ...


@runtime_checkable
class Redactor(Protocol):
    """Protocol for scrubbing sensitive identifiers from summaries."""

    def redact(self, text: str) -> str:
        """Scrubs sensitive entities, addresses, and identifiers from text."""
        ...


__all__ = ["Tokenizer", "Retriever", "Summarizer", "Redactor"]
