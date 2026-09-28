"""Layer 1: In-place head/tail message truncator.

Accepts raw turns. If a turn's tool output (or oversized content) exceeds
the cap threshold (default 35 tokens), truncates it down to:
head (22 tok) + marker ([... N tokens elided ...]) + tail (8 tok).
Preserves the raw uncapped content for downstream uncapped retrieval,
and returns the processed turns without evicting any turn yet.
"""

import copy

from context_engineer.config import ContextConfig
from context_engineer.tokenizer import count_tokens, slice_head_tail
from context_engineer.types import Turn


def cap_turn(turn: Turn, config: ContextConfig) -> Turn:
    """Caps oversized tool_output or content in a single Turn.

    Keeps raw_content and raw_tool_output intact for later retrieval.
    """
    # Clone turn to avoid unwanted in-place mutations of user's original object
    new_turn = copy.copy(turn)

    # Check tool_output first
    if new_turn.tool_output:
        tok_count = count_tokens(new_turn.tool_output, model_name=config.model_name)
        if tok_count > config.cap_threshold:
            new_turn.raw_tool_output = new_turn.tool_output
            new_turn.tool_output = slice_head_tail(
                new_turn.tool_output,
                head_tokens=config.cap_head_tokens,
                tail_tokens=config.cap_tail_tokens,
                model_name=config.model_name,
            )
            new_turn.is_capped = True

    # If the turn role is 'tool' or content itself is a huge tool log
    if (new_turn.role == "tool" or "[Tool Output]" in new_turn.content) and not new_turn.is_capped:
        tok_count = count_tokens(new_turn.content, model_name=config.model_name)
        if tok_count > config.cap_threshold:
            new_turn.raw_content = new_turn.content
            new_turn.content = slice_head_tail(
                new_turn.content,
                head_tokens=config.cap_head_tokens,
                tail_tokens=config.cap_tail_tokens,
                model_name=config.model_name,
            )
            new_turn.is_capped = True

    return new_turn


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
