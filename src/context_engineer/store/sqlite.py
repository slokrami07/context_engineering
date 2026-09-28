"""Zero-config SQLite conversation store."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from context_engineer.store.base import BaseConversationStore
from context_engineer.types import IngestionPayload, Turn


class SQLiteStore(BaseConversationStore):
    """Zero-config SQLite persistence layer for conversations and turns."""

    def __init__(self, db_path: str = "context_engineer.db") -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes database schema if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT,
                    turn_id INTEGER,
                    role TEXT,
                    content TEXT,
                    tool_output TEXT,
                    pinned INTEGER DEFAULT 0,
                    raw_content TEXT,
                    raw_tool_output TEXT,
                    is_capped INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    PRIMARY KEY (session_id, turn_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def save_turn(self, session_id: str, turn: Turn) -> None:
        """Saves or updates a single turn in a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure session entry exists
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, metadata_json) VALUES (?, ?)",
                (session_id, json.dumps({})),
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO turns (
                    session_id, turn_id, role, content, tool_output, pinned,
                    raw_content, raw_tool_output, is_capped, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn.id,
                    turn.role,
                    turn.content,
                    turn.tool_output,
                    1 if turn.pinned else 0,
                    turn.raw_content,
                    turn.raw_tool_output,
                    1 if turn.is_capped else 0,
                    json.dumps(turn.metadata),
                ),
            )
            conn.commit()

    def get_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Retrieves turns for a given session sorted by turn_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id ASC"
            params: list[object] = [session_id]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            turns: list[Turn] = []
            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                turn = Turn(
                    id=r["turn_id"],
                    role=r["role"],
                    content=r["content"],
                    tool_output=r["tool_output"],
                    pinned=bool(r["pinned"]),
                    raw_content=r["raw_content"],
                    raw_tool_output=r["raw_tool_output"],
                    is_capped=bool(r["is_capped"]),
                    metadata=meta,
                )
                turns.append(turn)
            return turns

    def save_session(self, payload: IngestionPayload) -> None:
        """Saves an entire IngestionPayload with all turns in a transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, system_prompt, metadata_json)
                VALUES (?, ?, ?)
                """,
                (
                    payload.session_id,
                    payload.system_prompt,
                    json.dumps(payload.metadata),
                ),
            )
            for turn in payload.turns:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO turns (
                        session_id, turn_id, role, content, tool_output, pinned,
                        raw_content, raw_tool_output, is_capped, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload.session_id,
                        turn.id,
                        turn.role,
                        turn.content,
                        turn.tool_output,
                        1 if turn.pinned else 0,
                        turn.raw_content,
                        turn.raw_tool_output,
                        1 if turn.is_capped else 0,
                        json.dumps(turn.metadata),
                    ),
                )
            conn.commit()

    def get_session(self, session_id: str) -> IngestionPayload | None:
        """Retrieves complete session with system prompt, metadata, and turns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                return None

            turns = self.get_turns(session_id)
            metadata = (
                json.loads(session_row["metadata_json"]) if session_row["metadata_json"] else {}
            )
            return IngestionPayload(
                session_id=session_id,
                turns=turns,
                system_prompt=session_row["system_prompt"],
                metadata=metadata,
            )

    def pin_turn(self, session_id: str, turn_id: int, pinned: bool = True) -> None:
        """Toggles the pinned status of a turn."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE turns SET pinned = ? WHERE session_id = ? AND turn_id = ?",
                (1 if pinned else 0, session_id, turn_id),
            )
            conn.commit()

    def list_sessions(self) -> list[str]:
        """Lists all session IDs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions ORDER BY created_at DESC")
            return [row["session_id"] for row in cursor.fetchall()]

    def delete_session(self, session_id: str) -> None:
        """Deletes session and all its turns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
