"""Unified rendering and final template gate (Algorithm A5).

Eliminates mid-conversation system messages, guarantees structural validity,
and measures rendered prompt tokens with template overhead before returning.
"""

from collections.abc import Sequence

from context_engineer.config import ContextConfig
from context_engineer.errors import StructuralError
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


def to_message(turn: Turn) -> Message:
    """Converts a domain Turn into a ChatML/provider Message with appropriate roles and IDs."""
    if turn.kind == "tool_call":
        return Message(
            role="assistant",
            content=turn.content,
            tool_calls=turn.tool_calls,
        )
    if turn.kind == "tool_result" or turn.role == "tool":
        return Message(
            role="tool",
            content=turn.content or turn.tool_output or "",
            tool_call_id=turn.tool_call_id,
            name=turn.name,
        )
    return Message(
        role=turn.role,
        content=turn.get_effective_text(),
    )


def merge_consecutive_roles(messages: Sequence[Message]) -> list[Message]:
    """Merges consecutive messages sharing identical roles to satisfy strict template alternation.

    Excludes tool results and messages containing tool calls from merging.
    """
    if not messages:
        return []

    merged: list[Message] = []
    for msg in messages:
        if not merged:
            merged.append(msg)
            continue

        prev = merged[-1]
        can_merge = (
            prev.role == msg.role
            and prev.role in ("user", "assistant", "system")
            and not prev.tool_calls
            and not msg.tool_calls
            and prev.tool_call_id is None
            and msg.tool_call_id is None
        )

        if can_merge:
            combined_content = f"{prev.content}\n\n{msg.content}" if prev.content else msg.content
            merged[-1] = Message(role=prev.role, content=combined_content)
        else:
            merged.append(msg)

    return merged


def validate_messages(
    messages: Sequence[Message],
    backend: str | None = None,
) -> None:
    """Validates structural integrity, role sequences, and tool call/result pairing (Algorithm A4).

    Raises:
        StructuralError: If messages violate template invariants.
    """
    pending_tool_call_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        # 1. Track tool calls and verify tool result pairing
        if msg.tool_calls:
            for call in msg.tool_calls:
                pending_tool_call_ids.add(call.id)

        if msg.role == "tool" or msg.tool_call_id:
            if not msg.tool_call_id or msg.tool_call_id not in pending_tool_call_ids:
                # If tool_call_id is None or unmatched, verify if preceding message was assistant tool_call
                prev = messages[idx - 1] if idx > 0 else None
                prev_is_call = prev and (prev.role == "assistant" and bool(prev.tool_calls))
                if not prev_is_call and msg.tool_call_id not in pending_tool_call_ids:
                    raise StructuralError(
                        f"Orphaned tool result at message index {idx} with tool_call_id={msg.tool_call_id!r}"
                    )
            if msg.tool_call_id in pending_tool_call_ids:
                pending_tool_call_ids.remove(msg.tool_call_id)

        # 2. Check for unexpected mid-dialogue system messages
        if idx > 0 and msg.role == "system":
            raise StructuralError(
                f"Unexpected mid-conversation system message at index {idx}. "
                "System messages must be fused at the conversation head."
            )

        # 3. Check for consecutive same-role text messages (if not tool)
        if idx > 0:
            prev = messages[idx - 1]
            if (
                prev.role == msg.role
                and prev.role in ("user", "system")
                and not prev.tool_calls
                and not msg.tool_calls
            ):
                raise StructuralError(
                    f"Consecutive messages with identical role {msg.role!r} at indices {idx - 1} and {idx}"
                )


def render(
    system_prompt: str,
    pinned: Sequence[Turn],
    summary: str,
    window: Sequence[Turn],
    hits: Sequence[Turn],
    query: str,
    config: ContextConfig,
) -> list[Message]:
    """Renders structured context messages into unified prefix and suffix zones (Algorithm A5).

    Prefix zone (stable):
        [system + pinned + summary] [window turns]
    Suffix zone (dynamic):
        [retrieved hits verbatim + user query]
    """
    # 1. Assemble single unified head system message
    head_parts: list[str] = []
    if system_prompt:
        head_parts.append(system_prompt.strip())

    if pinned:
        pinned_facts = "\n".join(f"- {p.get_effective_text()}" for p in pinned)
        head_parts.append(f"Pinned facts:\n{pinned_facts}")

    if summary and config.summary_placement in ("fused", "system"):
        head_parts.append(f"Earlier conversation (summary):\n{summary.strip()}")

    head_system_message = Message(
        role="system",
        content="\n\n".join(head_parts),
    )
    msgs: list[Message] = [head_system_message]

    # 2. Window turns
    msgs.extend(to_message(t) for t in window)

    # 3. Dynamic suffix (retrieved block + query as ONE user message)
    suffix_parts: list[str] = []
    if hits:
        # Chronological order
        sorted_hits = sorted(hits, key=lambda t: t.id)
        formatted_hits = "\n---\n".join(
            f"[{hit.role.upper()}]: {hit.get_effective_text()}" for hit in sorted_hits
        )
        suffix_parts.append(f"Relevant earlier context (verbatim):\n{formatted_hits}")

    if query:
        suffix_parts.append(query.strip())

    suffix_content = "\n\n".join(suffix_parts)
    msgs.append(Message(role="user", content=suffix_content))

    if config.merge_consecutive_roles:
        msgs = merge_consecutive_roles(msgs)

    return msgs


def gate(
    messages: Sequence[Message],
    tokenizer: Tokenizer,
    effective_ceiling: int,
) -> int:
    """Final gate measuring rendered prompt tokens with chat template overhead (Algorithm A5).

    Returns:
        Total measured tokens for prompt with generation priming.
    """
    return tokenizer.count_messages(messages, add_generation_prompt=True)
