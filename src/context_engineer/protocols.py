"""Protocols defining standard interfaces for tokenizers, retrievers, summarizers, redactors, stores, and backends."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from context_engineer.tokenizers.base import Tokenizer
from context_engineer.types import IngestionPayload, Message, Turn


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


@runtime_checkable
class DenseEmbedder(Protocol):
    """Protocol for dense vector embedding generation."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a sequence of text strings."""
        ...


@runtime_checkable
class Store(Protocol):
    """Protocol defining conversational history and session storage operations."""

    def save_turn(self, session_id: str, turn: Turn) -> None:
        """Saves or updates an individual turn in a session."""
        ...

    def get_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Retrieves turns for a given session in chronological sequence."""
        ...

    def save_session(self, payload: IngestionPayload) -> None:
        """Persists a complete IngestionPayload session."""
        ...

    def get_session(self, session_id: str) -> IngestionPayload | None:
        """Retrieves an entire IngestionPayload for a session ID."""
        ...

    def pin_turn(self, session_id: str, turn_id: int, pinned: bool = True) -> None:
        """Toggles the pinned status of a specific turn."""
        ...

    def list_sessions(self) -> list[str]:
        """Lists all existing session identifiers."""
        ...

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and its associated turns."""
        ...


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Cache statistics reported by inference backends."""

    measured: bool = False
    cached_tokens: int = 0
    total_prompt_tokens: int = 0
    cache_hit_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Standardized response from an inference backend."""

    content: str
    finish_reason: str = "stop"
    cache: CacheStats | None = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Protocol for inference runtime backends."""

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        stream: bool = False,
        **kw: Any,
    ) -> ChatResult:
        """Dispatches chat completion request to inference runtime."""
        ...


__all__ = [
    "Tokenizer",
    "Retriever",
    "Summarizer",
    "Redactor",
    "DenseEmbedder",
    "Store",
    "Backend",
    "CacheStats",
    "ChatResult",
]
