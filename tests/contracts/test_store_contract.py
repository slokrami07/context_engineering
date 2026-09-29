"""Contract tests for Store implementations."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from context_engineer.protocols import Store
from context_engineer.store.memory import InMemoryStore
from context_engineer.store.sqlite import SqliteStore
from context_engineer.types import IngestionPayload, Turn


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest) -> Generator[Store, None, None]:
    if request.param == "memory":
        yield InMemoryStore()
    elif request.param == "sqlite":
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_store.db"
            yield SqliteStore(db_path=str(db_path))
    else:
        pytest.skip(f"Unknown store {request.param}")


def test_store_protocol_conformance(store: Store) -> None:
    """Verifies that the store conforms to the Store protocol."""
    assert isinstance(store, Store)


def test_store_save_and_retrieve_turn(store: Store) -> None:
    """Verifies saving and retrieving individual turns in order."""
    session_id = "test-session-1"
    turn1 = Turn(id=1, role="user", content="First user query")
    turn2 = Turn(id=2, role="assistant", content="First assistant answer")

    store.save_turn(session_id, turn1)
    store.save_turn(session_id, turn2)

    retrieved = store.get_turns(session_id)
    assert len(retrieved) == 2
    assert retrieved[0].id == 1
    assert retrieved[0].content == "First user query"
    assert retrieved[1].id == 2
    assert retrieved[1].content == "First assistant answer"


def test_store_pin_turn(store: Store) -> None:
    """Verifies toggling pinned status on a turn."""
    session_id = "test-session-pin"
    turn = Turn(id=10, role="user", content="System invariant fact", pinned=False)
    store.save_turn(session_id, turn)

    turns_before = store.get_turns(session_id)
    assert turns_before[0].pinned is False

    store.pin_turn(session_id, 10, pinned=True)
    turns_after = store.get_turns(session_id)
    assert turns_after[0].pinned is True


def test_store_session_round_trip(store: Store) -> None:
    """Verifies full IngestionPayload round-trip."""
    session_id = "test-session-payload"
    payload = IngestionPayload(
        session_id=session_id,
        turns=[
            Turn(id=1, role="user", content="Initial greeting"),
            Turn(id=2, role="assistant", content="Hello there!"),
        ],
        system_prompt="Custom system instructions",
        metadata={"user_tier": "enterprise"},
    )

    store.save_session(payload)
    retrieved = store.get_session(session_id)

    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert len(retrieved.turns) == 2
    assert retrieved.system_prompt == "Custom system instructions"
    assert retrieved.metadata.get("user_tier") == "enterprise"


def test_store_list_and_delete_session(store: Store) -> None:
    """Verifies listing and deleting sessions."""
    store.save_turn("sess-a", Turn(id=1, role="user", content="A"))
    store.save_turn("sess-b", Turn(id=1, role="user", content="B"))

    sessions = store.list_sessions()
    assert "sess-a" in sessions
    assert "sess-b" in sessions

    store.delete_session("sess-a")
    remaining = store.list_sessions()
    assert "sess-a" not in remaining
    assert store.get_turns("sess-a") == []
