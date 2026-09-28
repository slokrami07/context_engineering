"""Unit tests for the 5 individual pipeline layers."""

from context_engineer.config import ContextConfig
from context_engineer.layers.cap import apply_cap_layer
from context_engineer.layers.pin import apply_pin_layer
from context_engineer.layers.retrieve import apply_retrieve_layer
from context_engineer.layers.summarize import apply_summarize_layer, scrub_entities
from context_engineer.layers.window import apply_window_layer
from context_engineer.types import Turn


def test_layer1_cap() -> None:
    config = ContextConfig(cap_threshold=35, cap_head_tokens=22, cap_tail_tokens=8)
    long_tool_output = "entry " * 80

    turn = Turn(
        id=1,
        role="assistant",
        content="Ran diagnostics.",
        tool_output=long_tool_output,
    )

    capped_turns = apply_cap_layer([turn], config)
    assert len(capped_turns) == 1
    capped = capped_turns[0]

    assert capped.is_capped is True
    assert capped.raw_tool_output == long_tool_output
    assert capped.tool_output is not None
    assert "[..." in capped.tool_output
    assert "tokens elided ...]" in capped.tool_output


def test_layer2_pin() -> None:
    config = ContextConfig(context_ceiling=4096, completion_reserve=256)
    turns = [
        Turn(id=0, role="user", content="System invariant rule 1", pinned=True),
        Turn(id=1, role="assistant", content="Normal response 1", pinned=False),
        Turn(id=2, role="user", content="System invariant rule 2", pinned=True),
    ]

    result = apply_pin_layer(turns, system_prompt="You are an agent.", config=config)
    assert len(result.pinned_turns) == 2
    assert len(result.unpinned_turns) == 1
    assert result.system_tokens > 0
    assert result.pinned_tokens > 0
    assert result.reserved_tokens == result.system_tokens + result.pinned_tokens
    assert result.remaining_budget == config.effective_budget - result.reserved_tokens


def test_layer3_retrieve_dropped_turn() -> None:
    config = ContextConfig(context_ceiling=400, completion_reserve=50, bm25_min_score=0.1)
    # Create turns where older turn contains needle
    turns = [
        Turn(
            id=0,
            role="assistant",
            content="Telemetry: shard-42 corrupted buffer.",
            tool_output="raw error dump",
        ),
    ]
    # Add many turns so turn 0 will be outside the small window budget
    for i in range(1, 20):
        turns.append(
            Turn(id=i, role="user", content=f"Step {i} regular discussion about cluster metrics.")
        )

    result = apply_retrieve_layer(
        unpinned_turns=turns,
        query="Which shard corrupted the buffer?",
        available_budget=150,
        config=config,
    )

    assert result.retrieved_turn is not None
    assert result.retrieved_turn.id == 0
    assert "shard-42" in result.retrieved_turn.content
    assert result.retrieval_tokens > 0


def test_layer4_window_contiguous() -> None:
    config = ContextConfig()
    turns = [Turn(id=i, role="user", content=f"Turn content {i}") for i in range(10)]

    # Budget that can fit ~3 turns
    result = apply_window_layer(unpinned_turns=turns, available_budget=80, config=config)

    assert len(result.window_turns) > 0
    # Must be contiguous from the tail
    last_id = turns[-1].id
    assert result.window_turns[-1].id == last_id
    # Window IDs should be sequential
    window_ids = [t.id for t in result.window_turns]
    for i in range(len(window_ids) - 1):
        assert window_ids[i + 1] == window_ids[i] + 1


def test_layer5_summarize_entity_scrubbing() -> None:
    raw_text = (
        "Investigation on shard-19 and partition-42 at 192.168.1.100 "
        "failed with error 0xDEADBEEF and commit 7f8a9b0c1d2e."
    )
    scrubbed = scrub_entities(raw_text)

    assert "shard-19" not in scrubbed
    assert "partition-42" not in scrubbed
    assert "192.168.1.100" not in scrubbed
    assert "0xDEADBEEF" not in scrubbed
    assert "7f8a9b0c1d2e" not in scrubbed
    assert "[resource]" in scrubbed
    assert "[code]" in scrubbed


def test_layer5_summarize_generates_narrative() -> None:
    config = ContextConfig(max_summary_tokens=100)
    dropped = [
        Turn(id=1, role="user", content="Check health of partition-99"),
        Turn(id=2, role="assistant", content="Running diagnostics on node-3"),
    ]

    summary, tokens = apply_summarize_layer(dropped, config=config)
    assert summary is not None
    assert "partition-99" not in summary
    assert "node-3" not in summary
    assert tokens > 0
