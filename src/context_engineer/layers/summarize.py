"""Layer 5: Entity-free abstract narrative summarizer for evicted turns (Algorithm A6).

Extracts salient discussion objectives without leaking domain-specific boilerplate or turn counts.
Applies single-pass redaction to scrub ephemeral identifiers and prevent stale fact hallucination.
"""

from collections.abc import Sequence

from context_engineer.config import ContextConfig
from context_engineer.redaction.redactor import RegexRedactor
from context_engineer.summarizers.extractive import ExtractiveSummarizer
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer
from context_engineer.types import Turn


def scrub_entities(
    text: str,
    mode: str = "light",
    resource_prefixes: list[str] | None = None,
) -> str:
    """Scrubs sensitive entities, addresses, and identifiers using RegexRedactor."""
    if mode not in ("none", "light", "strict"):
        mode = "light"
    redactor = RegexRedactor(mode=mode, resource_prefixes=resource_prefixes)  # type: ignore[arg-type]
    return redactor.redact(text)


def generate_narrative_summary(
    turns: Sequence[Turn],
    max_tokens: int = 256,
    tokenizer: Tokenizer | None = None,
    redaction_mode: str = "light",
    resource_prefixes: list[str] | None = None,
) -> str:
    """Generates a domain-agnostic extractive narrative summary for evicted turns."""
    if not turns or max_tokens <= 0:
        return ""

    tok = tokenizer or resolve_tokenizer("approx")
    redactor = RegexRedactor(mode=redaction_mode, resource_prefixes=resource_prefixes)  # type: ignore[arg-type]
    summarizer = ExtractiveSummarizer(redactor=redactor)
    return summarizer.summarize(turns, max_tokens=max_tokens, tok=tok)


def apply_summarize_layer(
    dropped_turns: Sequence[Turn],
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> tuple[str | None, int]:
    """Layer 5 pipeline entry point: generates abstract narrative summary for evicted turns.

    Args:
        dropped_turns: List of older turns not included in the recency window.
        config: Central configuration instance.
        tokenizer: Tokenizer instance.

    Returns:
        A tuple of (summary_text, summary_tokens). If dropped_turns is empty, returns (None, 0).
    """
    if config is None:
        config = ContextConfig()

    if not dropped_turns:
        return None, 0

    tok = tokenizer or resolve_tokenizer(config.tokenizer)
    summary = generate_narrative_summary(
        dropped_turns,
        max_tokens=config.max_summary_tokens,
        tokenizer=tok,
        redaction_mode=config.redaction_mode,
    )

    if not summary:
        return None, 0

    summary_tokens = tok.count(summary)
    return summary, summary_tokens
