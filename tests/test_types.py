import dataclasses

import pytest

from context_engineer.errors import HistoryError
from context_engineer.types import (
    AssembledContext,
    BudgetPlan,
    Message,
    ToolCall,
    Turn,
    validate_history,
)


def test_turn_frozen_immutability() -> None:
    turn = Turn(id=1, role="user", content="Hello world")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        turn.content = "Changed"  # type: ignore[misc]


def test_turn_metadata_defensive_copying() -> None:
    source_meta = {"key": "original_value"}
    turn = Turn(id=1, role="user", content="Hello", metadata=source_meta)

    # Mutating source dictionary should NOT affect the turn's internal metadata
    source_meta["key"] = "mutated_value"
    assert turn.metadata["key"] == "original_value"


def test_turn_content_hash_determinism_and_sensitivity() -> None:
    t1 = Turn(id=1, role="user", content="Same content")
    t2 = Turn(id=1, role="user", content="Same content")
    t3 = Turn(id=2, role="user", content="Same content")
    t4 = Turn(id=1, role="assistant", content="Same content")

    assert t1.content_hash == t2.content_hash
    assert t1.content_hash != t3.content_hash
    assert t1.content_hash != t4.content_hash


def test_legacy_turn_tool_output_emits_deprecation_warning() -> None:
    with pytest.deprecated_call(match="Turn.tool_output is deprecated"):
        turn = Turn(id=1, role="assistant", content="Diagnosed", tool_output="raw error log")

    assert turn.tool_output == "raw error log"
    assert "[Tool Output]: raw error log" in turn.get_effective_text()


def test_validate_history_monotonicity_and_uniqueness() -> None:
    # Valid history passes
    valid_history = [
        Turn(id=0, role="user", content="Start"),
        Turn(id=1, role="assistant", content="Response"),
        Turn(id=2, role="user", content="Next"),
    ]
    validate_history(valid_history)

    # Duplicate IDs raise HistoryError
    dup_history = [
        Turn(id=0, role="user", content="A"),
        Turn(id=0, role="assistant", content="B"),
    ]
    with pytest.raises(HistoryError, match="duplicate turn id 0"):
        validate_history(dup_history)

    # Non-monotonic IDs raise HistoryError
    non_monotonic = [
        Turn(id=0, role="user", content="A"),
        Turn(id=3, role="assistant", content="B"),
        Turn(id=2, role="user", content="C"),
    ]
    with pytest.raises(HistoryError, match="non-monotonic turn id"):
        validate_history(non_monotonic)


def test_tool_call_and_message_structure() -> None:
    tc = ToolCall(id="call_abc123", name="query_db", arguments='{"query": "SELECT 1"}')
    msg = Message(role="assistant", content="", tool_calls=(tc,))
    payload = msg.to_dict()

    assert payload["role"] == "assistant"
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["id"] == "call_abc123"
    assert payload["tool_calls"][0]["function"]["name"] == "query_db"


def test_assembled_context_retrieved_turn_shim() -> None:
    t = Turn(id=42, role="assistant", content="Needle fact")
    bp = BudgetPlan(
        context_ceiling=4096,
        completion_reserve=256,
        effective_budget=3840,
        system_tokens=20,
        pinned_tokens=10,
        summary_tokens=30,
        window_tokens=100,
        retrieval_tokens=50,
        query_tokens=15,
        total_used_tokens=225,
        remaining_tokens=3615,
    )

    assembled = AssembledContext(
        messages=[Message(role="user", content="Hi")],
        budget_plan=bp,
        ribbon_representation="[sys: 20t] ...",
        window_turns=[],
        retrieved_turns=(t,),
    )

    # Deprecated single retrieved_turn property is automatically populated
    assert assembled.retrieved_turn is not None
    assert assembled.retrieved_turn.id == 42
