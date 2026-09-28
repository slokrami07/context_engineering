"""Token counting and text slicing with ChatML template overhead awareness.

Provides backward-compatible wrappers forwarding to the hardened context_engineer.tokenizers subpackage.
"""

from typing import Any

from context_engineer.tokenizers import (
    Tokenizer,
    resolve_tokenizer,
)
from context_engineer.tokenizers import (
    slice_head_tail as _slice_head_tail,
)
from context_engineer.tokenizers import (
    truncate_text_to_tokens as _truncate_text_to_tokens,
)
from context_engineer.types import Message

# Backward compatibility alias
BaseTokenizerWrapper = Tokenizer

CHATML_MESSAGE_OVERHEAD = 4
CHATML_PROMPT_PRIMING = 3


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> Tokenizer:
    """Retrieves or initializes the tokenizer for the given model or spec."""
    return resolve_tokenizer(model_name)


def count_tokens(text: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> int:
    """Accurately counts raw string tokens."""
    if not text:
        return 0
    return get_tokenizer(model_name).count(text)


def count_message_tokens(
    role: str,
    content: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> int:
    """Counts tokens of a single ChatML formatted message including header and delimiter overhead."""
    tok = get_tokenizer(model_name)
    # Count message tokens without trailing generation prompt priming
    return tok.count_messages([Message(role=role, content=content)], add_generation_prompt=False)


def count_turn_tokens(turn: Any, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> int:
    """Counts tokens for an entire Turn object including content and tool outputs."""
    text = turn.get_effective_text()
    return count_message_tokens(turn.role, text, model_name=model_name)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    from_tail: bool = False,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> str:
    """Truncates text so that its token count does not exceed max_tokens."""
    return _truncate_text_to_tokens(
        text,
        max_tokens=max_tokens,
        tokenizer=get_tokenizer(model_name),
        from_tail=from_tail,
    )


def slice_head_tail(
    text: str,
    head_tokens: int,
    tail_tokens: int,
    marker_format: str = "[... {elided_count} tokens elided ...]",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> str:
    """Slices text into head + elision marker + tail tokens."""
    return _slice_head_tail(
        text,
        head_tokens=head_tokens,
        tail_tokens=tail_tokens,
        tokenizer=get_tokenizer(model_name),
        marker_format=marker_format,
    )
