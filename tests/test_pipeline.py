"""Unit tests for the master pipeline and prefix-cache ribbon ordering."""

from context_engineer.config import ContextConfig
from context_engineer.pipeline import assemble_context
from context_engineer.types import Turn


def test_ribbon_structure_and_order() -> None:
    config = ContextConfig(context_ceiling=600, completion_reserve=100)
    system_prompt = "You are a database engineer."

    turns = [
        Turn(id=0, role="user", content="System baseline config.", pinned=True),
        Turn(
            id=1,
            role="assistant",
            content="Found needle: partition-99 culprit.",
            tool_output="raw stack trace",
        ),
    ]
    # Add enough turns to exceed budget so dropped turns trigger summary generation
    for i in range(2, 60):
        turns.append(
            Turn(
                id=i,
                role="user",
                content=f"Step {i} normal log observation with verbose details about telemetry and metrics.",
            )
        )

    query = "Which partition was the culprit?"
    assembled = assemble_context(
        turns=turns,
        query=query,
        system_prompt=system_prompt,
        config=config,
    )

    msgs = assembled.messages
    assert len(msgs) >= 2

    # 1. Unified Head System message contains system prompt, pinned facts, and summary
    assert msgs[0].role == "system"
    assert system_prompt in msgs[0].content
    assert "Pinned facts:" in msgs[0].content
    assert "System baseline config." in msgs[0].content
    assert "Earlier conversation (summary):" in msgs[0].content

    # 2. Dynamic Suffix User message contains verbatim retrieved context and user query
    assert msgs[-1].role == "user"
    assert "Relevant earlier context (verbatim):" in msgs[-1].content
    assert "partition-99" in msgs[-1].content
    assert query in msgs[-1].content

    # Budget check
    assert assembled.budget_plan.is_valid
    assert assembled.budget_plan.total_used_tokens <= config.effective_budget


def test_prefix_cache_stability_at_same_depth() -> None:
    """Verifies that the left prefix is 100% identical across different queries."""
    config = ContextConfig(context_ceiling=800, completion_reserve=100)
    system_prompt = "System instructions."

    turns = [Turn(id=0, role="user", content="Pinned fact", pinned=True)]
    for i in range(1, 25):
        turns.append(Turn(id=i, role="user", content=f"Step {i} discussion."))

    # Two distinct queries asked at the same conversation depth
    ctx1 = assemble_context(
        turns=turns,
        query="Query Alpha: check telemetry?",
        system_prompt=system_prompt,
        config=config,
    )
    ctx2 = assemble_context(
        turns=turns,
        query="Query Beta: inspect memory pool?",
        system_prompt=system_prompt,
        config=config,
    )

    # Find the boundary: all messages before the retrieved block / question
    # The left side (system, pinned, summary, window turns) must match
    prefix_msgs1 = [
        m
        for m in ctx1.messages
        if not m.content.startswith("[Retrieved") and m != ctx1.messages[-1]
    ]
    prefix_msgs2 = [
        m
        for m in ctx2.messages
        if not m.content.startswith("[Retrieved") and m != ctx2.messages[-1]
    ]

    assert len(prefix_msgs1) == len(prefix_msgs2)
    for m1, m2 in zip(prefix_msgs1, prefix_msgs2, strict=True):
        assert m1.role == m2.role
        assert m1.content == m2.content
