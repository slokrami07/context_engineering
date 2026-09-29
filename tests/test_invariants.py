"""Property-based tests verifying core invariants I1 through I10 using Hypothesis."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_engineer.config import ContextConfig, TurnCapRule
from context_engineer.errors import ContextBudgetError, TokenizerError
from context_engineer.layers.cap import cap_turn
from context_engineer.pipeline import assemble_context
from context_engineer.tokenizers import resolve_tokenizer
from context_engineer.types import Turn


# Hypothesis strategy for valid Turn generation
@st.composite
def turn_strategy(draw: st.DrawFn, turn_id: int) -> Turn:
    role = draw(st.sampled_from(["user", "assistant"]))
    content = draw(
        st.text(min_size=5, max_size=200, alphabet=st.characters(blacklist_categories=["Cs"]))
    )
    pinned = draw(st.booleans()) if turn_id == 0 else False
    return Turn(id=turn_id, role=role, content=content, pinned=pinned)


@st.composite
def history_strategy(draw: st.DrawFn, min_size: int = 1, max_size: int = 25) -> list[Turn]:
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    turns: list[Turn] = []
    for i in range(size):
        # Enforce role alternation to ensure natural structural validity
        role = "user" if i % 2 == 0 else "assistant"
        content = draw(
            st.text(
                min_size=10, max_size=150, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 .,!?-"
            )
        )
        pinned = i == 0 and draw(st.booleans())
        turns.append(Turn(id=i, role=role, content=content, pinned=pinned))
    return turns


# -------------------------------------------------------------------------
# Invariant I1: Append-only stability
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=4, max_size=20))
@settings(max_examples=25, deadline=None)
def test_invariant_i1_append_only_stability(history: list[Turn]) -> None:
    """I1: Whenever the boundary is unchanged between t and t+1, the rendered prefix stream at t+1 starts with that at t."""
    config = ContextConfig(context_ceiling=2048, completion_reserve=128, window_chunk_tokens=100)
    tok = resolve_tokenizer(config.tokenizer)

    t_history = history[:-1]
    query = "Inspect system."

    res_t = assemble_context(
        t_history, query=query, system_prompt="Sys", config=config, tokenizer=tok
    )
    res_t_plus_1 = assemble_context(
        history, query=query, system_prompt="Sys", config=config, tokenizer=tok
    )

    boundary_t = res_t.report.window_start_index if res_t.report else 0
    boundary_t1 = res_t_plus_1.report.window_start_index if res_t_plus_1.report else 0

    if boundary_t == boundary_t1:
        # Boundary did not change: prefix messages of res_t must match head of res_t_plus_1 prefix messages
        prefix_t = [f"{m.role}:{m.content}" for m in res_t.messages[:-1]]
        prefix_t1 = [f"{m.role}:{m.content}" for m in res_t_plus_1.messages[:-1]]

        assert len(prefix_t1) >= len(prefix_t)
        assert prefix_t1[: len(prefix_t)] == prefix_t


# -------------------------------------------------------------------------
# Invariants I2 & I3: Query independence of prefix and window boundary
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=5, max_size=15))
@settings(max_examples=20, deadline=None)
def test_invariant_i2_i3_query_independence(history: list[Turn]) -> None:
    """I2/I3: The prefix zone messages and boundary are identical regardless of query."""
    config = ContextConfig(context_ceiling=2048, completion_reserve=128)
    tok = resolve_tokenizer(config.tokenizer)

    query_a = "First question regarding memory status."
    query_b = "Completely different query asking about database transactions."

    res_a = assemble_context(
        history, query=query_a, system_prompt="Sys", config=config, tokenizer=tok
    )
    res_b = assemble_context(
        history, query=query_b, system_prompt="Sys", config=config, tokenizer=tok
    )

    assert res_a.report is not None
    assert res_b.report is not None
    # Boundary must be identical
    assert res_a.report.window_start_index == res_b.report.window_start_index

    # Prefix messages must be byte-for-byte identical
    prefix_a = [(m.role, m.content) for m in res_a.messages[:-1]]
    prefix_b = [(m.role, m.content) for m in res_b.messages[:-1]]
    assert prefix_a == prefix_b

    # Prefix hashes must match
    assert res_a.report.prefix_hash == res_b.report.prefix_hash


# -------------------------------------------------------------------------
# Invariant I4: Monotonic boundary
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=8, max_size=20))
@settings(max_examples=20, deadline=None)
def test_invariant_i4_monotonic_boundary(history: list[Turn]) -> None:
    """I4: Window start index never decreases as history grows."""
    config = ContextConfig(context_ceiling=600, completion_reserve=100, window_chunk_tokens=50)
    tok = resolve_tokenizer(config.tokenizer)

    boundaries: list[int] = []
    for length in range(4, len(history) + 1):
        sub_history = history[:length]
        res = assemble_context(
            sub_history, query="status", system_prompt="Sys", config=config, tokenizer=tok
        )
        boundaries.append(res.report.window_start_index if res.report else 0)

    # Assert non-decreasing sequence
    for i in range(len(boundaries) - 1):
        assert boundaries[i + 1] >= boundaries[i], (
            f"Boundary decreased: {boundaries[i]} -> {boundaries[i + 1]}"
        )


# -------------------------------------------------------------------------
# Invariant I5: Budget safety
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=1, max_size=25))
@settings(max_examples=25, deadline=None)
def test_invariant_i5_budget_safety(history: list[Turn]) -> None:
    """I5: Rendered prompt tokens never exceed effective ceiling (or raises ContextBudgetError)."""
    config = ContextConfig(context_ceiling=1024, completion_reserve=128)
    tok = resolve_tokenizer(config.tokenizer)

    try:
        assembled = assemble_context(
            history, query="What is the state?", system_prompt="Sys", config=config, tokenizer=tok
        )
        rendered_count = tok.count_messages(assembled.messages, add_generation_prompt=True)
        assert rendered_count <= config.effective_budget
        assert assembled.budget_plan.is_valid
    except ContextBudgetError:
        # Loud failure is permitted when history/reserves cannot fit safely
        pass


# -------------------------------------------------------------------------
# Invariant I6: Structural validity
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=2, max_size=20))
@settings(max_examples=25, deadline=None)
def test_invariant_i6_structural_validity(history: list[Turn]) -> None:
    """I6: Window never starts on a tool result, pinned turns are preserved, and messages alternate validly."""
    config = ContextConfig(context_ceiling=2048, completion_reserve=256)
    tok = resolve_tokenizer(config.tokenizer)

    assembled = assemble_context(
        history, query="Check.", system_prompt="Sys", config=config, tokenizer=tok
    )

    # Window turns check
    if assembled.window_turns:
        assert assembled.window_turns[0].kind != "tool_result"

    # Pinned turns check: any pinned turn in history must be accounted for in assembled context
    pinned_in_input = [t for t in history if t.pinned]
    assert len(assembled.pinned_turns) == len(pinned_in_input)

    # Rendered messages check: first message is system, subsequent messages do not have adjacent duplicate text roles
    msgs = assembled.messages
    assert msgs[0].role == "system"
    for i in range(1, len(msgs) - 1):
        # Consecutive text messages cannot share identical role if merged
        assert not (msgs[i].role == msgs[i + 1].role and msgs[i].role in ("user", "system"))


# -------------------------------------------------------------------------
# Invariant I7: Determinism
# -------------------------------------------------------------------------
@given(history=history_strategy(min_size=3, max_size=15))
@settings(max_examples=20, deadline=None)
def test_invariant_i7_determinism(history: list[Turn]) -> None:
    """I7: Two assemblies with identical inputs produce identical messages and hashes."""
    config = ContextConfig(context_ceiling=2048, completion_reserve=256)
    tok = resolve_tokenizer(config.tokenizer)

    res1 = assemble_context(
        history, query="Query A", system_prompt="Sys", config=config, tokenizer=tok
    )
    res2 = assemble_context(
        history, query="Query A", system_prompt="Sys", config=config, tokenizer=tok
    )

    assert res1.messages == res2.messages
    assert res1.report is not None and res2.report is not None
    assert res1.report.prefix_hash == res2.report.prefix_hash
    assert res1.budget_plan == res2.budget_plan


# -------------------------------------------------------------------------
# Invariant I8: Pure, idempotent capping
# -------------------------------------------------------------------------
@given(
    content=st.text(min_size=10, max_size=1000, alphabet="abcdefghijklmnopqrstuvwxyz 0123456789")
)
@settings(max_examples=25, deadline=None)
def test_invariant_i8_idempotent_capping(content: str) -> None:
    """I8: cap(cap(t)) == cap(t), raw content preserved, tokens <= threshold."""
    config = ContextConfig(cap_threshold=35, cap_head_tokens=22, cap_tail_tokens=8)
    tok = resolve_tokenizer(config.tokenizer)

    turn = Turn(id=1, role="assistant", content="Summary", tool_output=content)

    cap1 = cap_turn(turn, config, tokenizer=tok)
    cap2 = cap_turn(cap1, config, tokenizer=tok)
    cap3 = cap_turn(cap2, config, tokenizer=tok)

    assert cap1 == cap2
    assert cap2 == cap3

    if cap1.is_capped:
        assert cap1.raw_tool_output == content
        assert tok.count(cap1.tool_output or "") <= 35


def test_invariant_i8_json_capping_validity() -> None:
    """I8 JSON strategy: output remains valid parseable JSON."""
    rule = TurnCapRule(threshold=80, head_tokens=40, tail_tokens=20, strategy="json")
    config = ContextConfig(cap_policy=ContextConfig().cap_policy.__class__(default_tool_rule=rule))
    tok = resolve_tokenizer(config.tokenizer)

    large_dict = {f"key_{i}": [f"val_{j}" for j in range(15)] for i in range(10)}
    raw_json = json.dumps(large_dict)

    turn = Turn(id=1, role="tool", content=raw_json)
    capped = cap_turn(turn, config, tokenizer=tok)

    assert capped.is_capped
    assert tok.count(capped.content) <= 80
    # Output must still parse as valid JSON
    parsed = json.loads(capped.content)
    assert isinstance(parsed, dict)


# -------------------------------------------------------------------------
# Invariant I9: Loud failure on reserve overflow
# -------------------------------------------------------------------------
def test_invariant_i9_loud_failure() -> None:
    """I9: ContextBudgetError raised when fixed reserves exceed effective ceiling."""
    config = ContextConfig(
        context_ceiling=600,
        completion_reserve=100,
        suffix_reserve_tokens=150,
        max_summary_tokens=100,
    )
    # Huge system prompt causes fixed prefix cost to overflow effective ceiling
    huge_system = "System directive " * 200
    with pytest.raises(ContextBudgetError):
        assemble_context([], query="test", system_prompt=huge_system, config=config)


# -------------------------------------------------------------------------
# Invariant I10: Offline security / no hidden side effects
# -------------------------------------------------------------------------
def test_invariant_i10_offline_security() -> None:
    """I10: strict_tokenizer=True raises TokenizerError instead of silent downloads."""
    with pytest.raises(TokenizerError):
        resolve_tokenizer("hf:non-existent/remote-repo", strict=True, allow_download=False)
