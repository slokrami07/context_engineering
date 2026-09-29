"""In-memory conversation store implementing the Store protocol."""

import threading
from collections import defaultdict

from context_engineer.store.base import BaseConversationStore
from context_engineer.types import IngestionPayload, Turn


class InMemoryStore(BaseConversationStore):
    """Thread-safe in-memory store for session turns and metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, IngestionPayload] = {}
        self._turns: dict[str, dict[int, Turn]] = defaultdict(dict)

    def save_turn(self, session_id: str, turn: Turn) -> None:
        """Saves or updates an individual turn in a session."""
        with self._lock:
            self._turns[session_id][turn.id] = turn
            if session_id in self._sessions:
                # Synchronize session turns
                sorted_turns = sorted(self._turns[session_id].values(), key=lambda t: t.id)
                self._sessions[session_id].turns = sorted_turns

    def get_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Retrieves turns for a given session in chronological sequence."""
        with self._lock:
            turns = sorted(self._turns.get(session_id, {}).values(), key=lambda t: t.id)
            if limit is not None:
                return turns[-limit:]
            return list(turns)

    def save_session(self, payload: IngestionPayload) -> None:
        """Persists a complete IngestionPayload session."""
        with self._lock:
            self._sessions[payload.session_id] = payload
            for turn in payload.turns:
                self._turns[payload.session_id][turn.id] = turn

    def get_session(self, session_id: str) -> IngestionPayload | None:
        """Retrieves an entire IngestionPayload for a session ID."""
        with self._lock:
            if session_id not in self._sessions:
                return None
            session = self._sessions[session_id]
            sorted_turns = sorted(self._turns.get(session_id, {}).values(), key=lambda t: t.id)
            return IngestionPayload(
                session_id=session.session_id,
                turns=sorted_turns,
                system_prompt=session.system_prompt,
                metadata=dict(session.metadata),
            )

    def pin_turn(self, session_id: str, turn_id: int, pinned: bool = True) -> None:
        """Toggles the pinned status of a specific turn."""
        with self._lock:
            if session_id in self._turns and turn_id in self._turns[session_id]:
                import dataclasses

                turn = self._turns[session_id][turn_id]
                updated = dataclasses.replace(turn, pinned=pinned)
                self._turns[session_id][turn_id] = updated
                if session_id in self._sessions:
                    sorted_turns = sorted(self._turns[session_id].values(), key=lambda t: t.id)
                    self._sessions[session_id].turns = sorted_turns

    def list_sessions(self) -> list[str]:
        """Lists all existing session identifiers."""
        with self._lock:
            return sorted(set(self._sessions.keys()) | set(self._turns.keys()))

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and its associated turns."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._turns.pop(session_id, None)
