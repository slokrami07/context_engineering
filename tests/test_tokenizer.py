"""Unit tests for ChatML tokenizer and token slicing utilities."""

import pytest
from context_engineer.tokenizer import (
    count_tokens,
    count_message_tokens,
    count_turn_tokens,
    slice_head_tail,
    truncate_text_to_tokens,
    get_tokenizer,
)
from context_engineer.types import Turn


def test_tokenizer_initialization() -> None:
    tok = get_tokenizer()
    assert tok is not None
    tokens = tok.encode("Hello world!")
    assert len(tokens) > 0
    decoded = tok.decode(tokens)
    assert "Hello world!" in decoded


def test_count_message_tokens_overhead() -> None:
    raw = count_tokens("test message")
    msg_tokens = count_message_tokens("user", "test message")
    # ChatML overhead should ensure msg_tokens > raw
    assert msg_tokens > raw


def test_slice_head_tail_preserves_bounds() -> None:
    long_text = "word " * 100
    sliced = slice_head_tail(long_text, head_tokens=22, tail_tokens=8)
    assert "[..." in sliced
    assert "tokens elided ...]" in sliced
    # Sliced text should be significantly shorter than original
    assert len(sliced) < len(long_text)


def test_slice_head_tail_short_text_unchanged() -> None:
    short_text = "short log"
    sliced = slice_head_tail(short_text, head_tokens=22, tail_tokens=8)
    assert sliced == short_text


def test_truncate_text_to_tokens() -> None:
    text = "The quick brown fox jumps over the lazy dog repeatedly and continuously."
    truncated = truncate_text_to_tokens(text, max_tokens=5)
    tok_count = count_tokens(truncated)
    assert tok_count <= 5


def test_count_turn_tokens_with_tool_output() -> None:
    turn = Turn(
        id=1,
        role="assistant",
        content="Inspecting database...",
        tool_output="Status: OK, Connection alive.",
    )
    tokens = count_turn_tokens(turn)
    assert tokens > count_tokens("Inspecting database...")
