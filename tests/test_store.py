"""Unit tests for SQLite conversation store."""

import os
import tempfile
from collections.abc import Generator

import pytest

from context_engineer.store.sqlite import SQLiteStore
from context_engineer.types import IngestionPayload, Turn


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_sqlite_save_and_retrieve_turn(temp_db_path: str) -> None:
    store = SQLiteStore(db_path=temp_db_path)

    turn = Turn(
        id=0,
        role="user",
        content="Testing SQLite turn",
        tool_output="Diagnostic OK",
        pinned=True,
    )
    store.save_turn("sess-1", turn)

    turns = store.get_turns("sess-1")
    assert len(turns) == 1
    retrieved = turns[0]
    assert retrieved.id == 0
    assert retrieved.role == "user"
    assert retrieved.content == "Testing SQLite turn"
    assert retrieved.tool_output == "Diagnostic OK"
    assert retrieved.pinned is True


def test_sqlite_pin_toggle(temp_db_path: str) -> None:
    store = SQLiteStore(db_path=temp_db_path)

    turn = Turn(id=1, role="assistant", content="Response to pin", pinned=False)
    store.save_turn("sess-2", turn)

    assert store.get_turns("sess-2")[0].pinned is False

    store.pin_turn("sess-2", turn_id=1, pinned=True)
    assert store.get_turns("sess-2")[0].pinned is True


def test_sqlite_save_and_get_session(temp_db_path: str) -> None:
    store = SQLiteStore(db_path=temp_db_path)

    payload = IngestionPayload(
        session_id="sess-full",
        system_prompt="Custom system prompt.",
        turns=[
            Turn(id=0, role="user", content="Hello"),
            Turn(id=1, role="assistant", content="Hi there"),
        ],
        metadata={"project": "context-engineer"},
    )
    store.save_session(payload)

    loaded = store.get_session("sess-full")
    assert loaded is not None
    assert loaded.session_id == "sess-full"
    assert loaded.system_prompt == "Custom system prompt."
    assert loaded.metadata.get("project") == "context-engineer"
    assert len(loaded.turns) == 2

    # Listing & Deletion
    assert "sess-full" in store.list_sessions()
    store.delete_session("sess-full")
    assert "sess-full" not in store.list_sessions()
