"""Abstract Base Store for persisting conversations, turns, and metadata."""

from abc import ABC, abstractmethod
from typing import Optional

from context_engineer.types import Turn, IngestionPayload


class BaseConversationStore(ABC):
    """Abstract base class defining conversation storage operations."""

    @abstractmethod
    def save_turn(self, session_id: str, turn: Turn) -> None:
        """Saves or updates an individual turn in a session."""
        pass

    @abstractmethod
    def get_turns(self, session_id: str, limit: Optional[int] = None) -> list[Turn]:
        """Retrieves turns for a given session in chronological sequence."""
        pass

    @abstractmethod
    def save_session(self, payload: IngestionPayload) -> None:
        """Persists a complete IngestionPayload session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[IngestionPayload]:
        """Retrieves an entire IngestionPayload for a session ID."""
        pass

    @abstractmethod
    def pin_turn(self, session_id: str, turn_id: int, pinned: bool = True) -> None:
        """Toggles the pinned status of a specific turn."""
        pass

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """Lists all existing session identifiers."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Deletes a session and its associated turns."""
        pass
