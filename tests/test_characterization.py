"""Characterization tests pinning baseline (v0.1.0) behavior of all 5 layers and the assembly pipeline.

These tests establish an immutable record of baseline behavior prior to Phase 1 & Phase 2 refactorings.
Any future intentional deviations from these pinned behaviors must be explicitly recorded in CHANGELOG.md.
"""

from context_engineer.config import ContextConfig
from context_engineer.layers.cap import cap_turn
from context_engineer.layers.pin import apply_pin_layer
from context_engineer.layers.retrieve import apply_retrieve_layer
from context_engineer.layers.summarize import (
    scrub_entities,
)
from context_engineer.layers.window import apply_window_layer
from context_engineer.pipeline import assemble_context
from context_engineer.types import Turn


def test_characterize_cap_layer_truncation() -> None:
    """Pin: Tool output > 35 tokens is truncated to head + tail with elision marker."""
    config = ContextConfig(cap_threshold=35, cap_head_tokens=22, cap_tail_tokens=8)
    long_log = "log_entry_datum " * 50

    turn = Turn(
        id=42,
        role="assistant",
        content="Inspecting subsystem.",
        tool_output=long_log,
    )

    capped = cap_turn(turn, config)
    assert capped.is_capped is True
    assert capped.raw_tool_output == long_log
    assert capped.tool_output is not None
    assert "[..." in capped.tool_output
    assert "tokens elided ...]" in capped.tool_output
    # Pinned: original content is unchanged when only tool_output exceeds threshold
    assert capped.content == "Inspecting subsystem."


def test_characterize_pin_layer_deduction() -> None:
    """Pin: Invariant facts and system prompt are deducted upfront from effective budget."""
    config = ContextConfig(context_ceiling=2000, completion_reserve=200)
    # effective_budget = 1800
    turns = [
        Turn(id=0, role="user", content="System baseline directive A", pinned=True),
        Turn(id=1, role="assistant", content="Dynamic conversational answer", pinned=False),
        Turn(id=2, role="user", content="System baseline directive B", pinned=True),
    ]

    pin_res = apply_pin_layer(turns, system_prompt="Baseline instructions", config=config)
    assert len(pin_res.pinned_turns) == 2
    assert len(pin_res.unpinned_turns) == 1
    assert pin_res.pinned_turns[0].id == 0
    assert pin_res.pinned_turns[1].id == 2
    assert pin_res.reserved_tokens == pin_res.system_tokens + pin_res.pinned_tokens
    assert pin_res.remaining_budget == config.effective_budget - pin_res.reserved_tokens


def test_characterize_window_layer_recency_backward_packing() -> None:
    """Pin: Window packs turns backwards from the latest turn to guarantee contiguous recency."""
    config = ContextConfig()
    turns = [Turn(id=i, role="user", content=f"Message {i}") for i in range(15)]

    # Budget sufficient for ~4 turns
    win_res = apply_window_layer(unpinned_turns=turns, available_budget=100, config=config)

    assert len(win_res.window_turns) > 0
    assert win_res.window_turns[-1].id == 14
    # All turns in window are strictly contiguous
    for i in range(len(win_res.window_turns) - 1):
        assert win_res.window_turns[i + 1].id == win_res.window_turns[i].id + 1
    # Dropped turns are all turns preceding the first window turn
    first_win_id = win_res.window_turns[0].id
    assert [t.id for t in win_res.dropped_turns] == list(range(first_win_id))


def test_characterize_retrieve_layer_bm25_uncapped_restoration() -> None:
    """Pin: Retrieval searches dropped turns using BM25 and restores uncapped content."""
    config = ContextConfig(bm25_min_score=0.1)
    needle = "shard-77 crashed due to segfault"

    # Pre-capped turn with raw_content preserved
    turn0 = Turn(
        id=0,
        role="assistant",
        content="shard-77 [... elided ...]",
        raw_content=needle,
        is_capped=True,
    )
    # Add turns that will fill the window budget so turn 0 is dropped
    turns = [turn0] + [
        Turn(id=i, role="user", content=f"Regular filler conversational turn {i}")
        for i in range(1, 20)
    ]

    ret_res = apply_retrieve_layer(
        unpinned_turns=turns,
        query="Why did shard-77 crash?",
        available_budget=120,
        config=config,
    )

    assert ret_res.retrieved_turn is not None
    assert ret_res.retrieved_turn_id == 0
    # Pin: uncapped text contains full original needle
    assert ret_res.retrieved_turn.is_capped is False
    assert needle in ret_res.retrieved_turn.content


def test_characterize_summarize_layer_entity_scrubbing() -> None:
    """Pin: Regex scrubs specific identifiers into abstract tokens."""
    dirty_text = (
        "Host worker-42 in cluster-west at 10.0.0.1 failed with 0xDEADBEEF "
        "on partition-88 commit a1b2c3d4e5f67890."
    )
    clean = scrub_entities(dirty_text)
    assert "[resource]" in clean
    assert "[address]" in clean
    assert "[code]" in clean
    assert "[hash]" in clean
    assert "worker-42" not in clean
    assert "10.0.0.1" not in clean
    assert "0xDEADBEEF" not in clean


def test_characterize_pipeline_ribbon_structure() -> None:
    """Pin: Assembled context ribbon strictly separates prefix-stable and dynamic suffix."""
    config = ContextConfig(context_ceiling=600, completion_reserve=100)
    system_prompt = "You are a test assistant."

    turns = [
        Turn(id=0, role="user", content="Pinned fact 1", pinned=True),
        Turn(id=1, role="assistant", content="Target needle: error code 0x1234"),
    ]
    for i in range(2, 60):
        turns.append(
            Turn(
                id=i,
                role="user",
                content=f"Step {i} normal log observation with verbose details about telemetry and metrics.",
            )
        )

    query = "What was the error code?"
    assembled = assemble_context(
        turns=turns,
        query=query,
        system_prompt=system_prompt,
        config=config,
    )

    # Message roles: [system, pinned(user), summary(system), window..., retrieved(system), query(user)]
    assert assembled.messages[0].role == "system"
    assert assembled.messages[0].content == system_prompt
    assert assembled.messages[1].content == "Pinned fact 1"
    # Penultimate message must be the retrieved block
    assert assembled.messages[-2].role == "system"
    assert "[Retrieved Relevant Historical Context]" in assembled.messages[-2].content
    # Last message is user query
    assert assembled.messages[-1].role == "user"
    assert assembled.messages[-1].content == query

    # Ribbon string format representation
    assert "[sys:" in assembled.ribbon_representation
    assert "[pin:" in assembled.ribbon_representation
    assert "[sum:" in assembled.ribbon_representation
    assert "| [ret:" in assembled.ribbon_representation
    assert "[q:" in assembled.ribbon_representation
