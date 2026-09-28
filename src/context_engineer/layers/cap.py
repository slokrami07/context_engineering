"""Layer 1: In-place head/tail message truncator.

Accepts raw turns. If a turn's tool output (or oversized content) exceeds
the cap threshold (default 35 tokens), truncates it down to:
head (22 tok) + marker ([... N tokens elided ...]) + tail (8 tok).
Preserves the raw uncapped content for downstream uncapped retrieval,
and returns the processed turns without evicting any turn yet.
"""

import dataclasses

from context_engineer.config import ContextConfig
from context_engineer.tokenizer import count_tokens, slice_head_tail
from context_engineer.types import Turn


def cap_turn(turn: Turn, config: ContextConfig) -> Turn:
    """Caps oversized tool_output or content in a single Turn.

    Keeps raw_content and raw_tool_output intact for later retrieval.
    """
    tool_output = turn.tool_output
    raw_tool_output = turn.raw_tool_output
    content = turn.content
    raw_content = turn.raw_content
    is_capped = turn.is_capped

    # Check tool_output first
    if tool_output:
        tok_count = count_tokens(tool_output, model_name=config.model_name)
        if tok_count > config.cap_threshold:
            if raw_tool_output is None:
                raw_tool_output = tool_output
            tool_output = slice_head_tail(
                tool_output,
                head_tokens=config.cap_head_tokens,
                tail_tokens=config.cap_tail_tokens,
                model_name=config.model_name,
            )
            is_capped = True

    # If the turn role is 'tool' or content itself is a huge tool log
    if (turn.role == "tool" or "[Tool Output]" in turn.content) and not is_capped:
        tok_count = count_tokens(content, model_name=config.model_name)
        if tok_count > config.cap_threshold:
            if raw_content is None:
                raw_content = content
            content = slice_head_tail(
                content,
                head_tokens=config.cap_head_tokens,
                tail_tokens=config.cap_tail_tokens,
                model_name=config.model_name,
            )
            is_capped = True

    if not is_capped and tool_output == turn.tool_output and content == turn.content:
        return turn

    return dataclasses.replace(
        turn,
        content=content,
        tool_output=tool_output,
        raw_content=raw_content,
        raw_tool_output=raw_tool_output,
        is_capped=is_capped,
    )


def apply_cap_layer(turns: list[Turn], config: ContextConfig | None = None) -> list[Turn]:
    """Layer 1 pipeline entry point: truncates oversized tool outputs across all turns.

    Args:
        turns: List of raw conversational turns.
        config: Central configuration instance (uses default if None).

    Returns:
        List of turns with capped tool outputs, retaining full turn count.
    """
    if config is None:
        config = ContextConfig()

    return [cap_turn(t, config) for t in turns]
