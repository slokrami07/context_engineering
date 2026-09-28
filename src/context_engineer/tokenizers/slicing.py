"""Safe, character-boundary aware token slicing and truncation utilities."""

from context_engineer.tokenizers.base import Tokenizer


def clean_utf8_decode(tokens: list[int], tokenizer: Tokenizer) -> str:
    """Decodes token IDs, adjusting boundaries to prevent broken multi-byte replacement characters."""
    if not tokens:
        return ""
    decoded = tokenizer.decode(tokens)
    # Check if replacement character U+FFFD appeared at boundary due to slicing a multi-byte byte-pair (K6)
    if "\ufffd" in decoded and len(tokens) > 1:
        # Try trimming 1 token from the broken edge
        trimmed_right = tokenizer.decode(tokens[:-1])
        if "\ufffd" not in trimmed_right:
            return trimmed_right
        trimmed_left = tokenizer.decode(tokens[1:])
        if "\ufffd" not in trimmed_left:
            return trimmed_left
    return decoded.replace("\ufffd", "")


def slice_head_tail(
    text: str,
    head_tokens: int,
    tail_tokens: int,
    tokenizer: Tokenizer,
    marker_format: str = "[... {elided_count} tokens elided ...]",
) -> str:
    """Slices a text into head tokens + elision marker + tail tokens with boundary and compression guarantees.

    Invariants enforced:
    1. tail_tokens must be >= 1 (preventing tail_tokens=0 slice returning entire sequence - K4).
    2. Capping only triggers if elided_count > marker_cost + 2, preventing negative compression (K5).
    3. Trims multi-byte slice boundaries to prevent U+FFFD artifacts (K6).
    4. Re-measures output to guarantee capped tokens do not exceed threshold.
    """
    if tail_tokens < 1:
        raise ValueError(f"tail_tokens must be >= 1, got {tail_tokens}")
    if head_tokens < 1:
        raise ValueError(f"head_tokens must be >= 1, got {head_tokens}")

    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= (head_tokens + tail_tokens):
        return text

    elided_count = total_tokens - (head_tokens + tail_tokens)
    marker = marker_format.format(elided_count=elided_count)
    marker_cost = tokenizer.count(marker)

    # Invariant K5: Negative compression check.
    # If the elision does not save at least (marker_cost + 2) tokens, keep original text
    if elided_count <= marker_cost + 2:
        return text

    head_tokens_slice = tokens[:head_tokens]
    tail_tokens_slice = tokens[-tail_tokens:]

    head_part = clean_utf8_decode(head_tokens_slice, tokenizer)
    tail_part = clean_utf8_decode(tail_tokens_slice, tokenizer)

    assembled = f"{head_part.rstrip()}\n{marker}\n{tail_part.lstrip()}"

    # Verify and enforce that the assembled text is strictly smaller than original
    assembled_count = tokenizer.count(assembled)
    if assembled_count >= total_tokens:
        return text

    return assembled


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Tokenizer,
    from_tail: bool = False,
) -> str:
    """Truncates text so that its token count does not exceed max_tokens without UTF-8 corruption."""
    if max_tokens <= 0 or not text:
        return ""

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    sliced = tokens[-max_tokens:] if from_tail else tokens[:max_tokens]
    return clean_utf8_decode(sliced, tokenizer)
