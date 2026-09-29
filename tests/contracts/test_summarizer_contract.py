"""Contract tests for Summarizer implementations."""

import pytest

from context_engineer.protocols import Summarizer
from context_engineer.summarizers.extractive import ExtractiveSummarizer
from context_engineer.tokenizers.approx import ApproxTokenizer
from context_engineer.types import Turn


@pytest.fixture
def summarizer() -> Summarizer:
    return ExtractiveSummarizer()


def test_summarizer_protocol_conformance(summarizer: Summarizer) -> None:
    """Verifies that the object conforms to the Summarizer protocol."""
    assert isinstance(summarizer, Summarizer)
    assert hasattr(summarizer, "id")
    assert hasattr(summarizer, "version")


def test_summarizer_empty_and_zero_budget(summarizer: Summarizer) -> None:
    """Verifies empty return on empty candidates or zero max_tokens."""
    tok = ApproxTokenizer()
    turns = [Turn(id=1, role="user", content="Test goal.")]
    assert summarizer.summarize([], max_tokens=100, tok=tok) == ""
    assert summarizer.summarize(turns, max_tokens=0, tok=tok) == ""


def test_summarizer_budget_adherence(summarizer: Summarizer) -> None:
    """Verifies that generated summary tokens do not exceed max_tokens."""
    tok = ApproxTokenizer()
    turns = [
        Turn(
            id=i,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn content {i} with long explanations.",
        )
        for i in range(15)
    ]
    max_tokens = 50
    summary = summarizer.summarize(turns, max_tokens=max_tokens, tok=tok)

    assert isinstance(summary, str)
    assert tok.count(summary) <= max_tokens
