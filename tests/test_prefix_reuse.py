"""Verification of prefix reuse across per-turn append progression (Acceptance Criterion 2)."""

from context_engineer.config import ContextConfig
from context_engineer.pipeline import assemble_context
from context_engineer.tokenizers import resolve_tokenizer
from context_engineer.types import Turn


def test_prefix_reuse_across_append_progression() -> None:
    """Verifies that prefix reuse between consecutive non-compaction turns is >= 90%."""
    config = ContextConfig(
        context_ceiling=4096,
        completion_reserve=256,
        window_chunk_tokens=200,
        suffix_reserve_tokens=500,
        max_summary_tokens=200,
    )
    tok = resolve_tokenizer(config.tokenizer)

    turns: list[Turn] = [
        Turn(id=0, role="user", content="Deploy system configuration version 2.4", pinned=True),
        Turn(
            id=1, role="assistant", content="Acknowledged. System configuration v2.4 initialized."
        ),
    ]

    system_prompt = "You are an autonomous systems assistant."
    non_compaction_reuse_rates: list[float] = []

    prev_prefix_msgs: list[str] | None = None
    prev_boundary: int | None = None

    for step in range(2, 40):
        # Alternate roles
        role = "user" if step % 2 == 0 else "assistant"
        content = (
            f"Step {step}: status inquiry regarding process {step * 10} and memory allocations."
        )
        turns.append(Turn(id=step, role=role, content=content))

        query = f"Provide summary of step {step}."
        assembled = assemble_context(
            turns=turns,
            query=query,
            system_prompt=system_prompt,
            config=config,
            tokenizer=tok,
        )

        curr_boundary = assembled.report.window_start_index if assembled.report else 0
        curr_prefix_msgs = [f"{m.role}:{m.content}" for m in assembled.messages[:-1]]

        if (
            prev_prefix_msgs is not None
            and prev_boundary is not None
            and curr_boundary == prev_boundary
        ):
            # Check if this is a non-compaction turn
            # Find length of common prefix messages
            common_len = 0
            for m_prev, m_curr in zip(prev_prefix_msgs, curr_prefix_msgs, strict=False):
                if m_prev == m_curr:
                    common_len += 1
                else:
                    break

            # Count tokens of common prefix vs previous prefix tokens
            common_msg_objs = assembled.messages[:common_len]
            prev_msg_objs = assembled.messages[: len(prev_prefix_msgs)]

            common_tokens = tok.count_messages(common_msg_objs, add_generation_prompt=False)
            prev_total_tokens = tok.count_messages(prev_msg_objs, add_generation_prompt=False)

            if prev_total_tokens > 0:
                reuse_ratio = common_tokens / prev_total_tokens
                non_compaction_reuse_rates.append(reuse_ratio)

        prev_prefix_msgs = curr_prefix_msgs
        prev_boundary = curr_boundary

    assert len(non_compaction_reuse_rates) > 0, "No non-compaction transitions observed"
    avg_reuse = sum(non_compaction_reuse_rates) / len(non_compaction_reuse_rates)
    min_reuse = min(non_compaction_reuse_rates)

    print(
        f"\n[Prefix Reuse Report]: Average Reuse = {avg_reuse * 100:.2f}%, Minimum Reuse = {min_reuse * 100:.2f}%"
    )
    assert avg_reuse >= 0.90, f"Average prefix reuse {avg_reuse:.2%} is below 90% requirement"
    assert min_reuse >= 0.90, f"Minimum prefix reuse {min_reuse:.2%} is below 90% requirement"
