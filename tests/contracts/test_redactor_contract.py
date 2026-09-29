"""Contract tests for Redactor implementations."""

import pytest

from context_engineer.protocols import Redactor
from context_engineer.redaction.redactor import RegexRedactor


@pytest.fixture
def redactor() -> Redactor:
    return RegexRedactor(mode="light")


def test_redactor_protocol_conformance(redactor: Redactor) -> None:
    """Verifies that the redactor conforms to the Redactor protocol."""
    assert isinstance(redactor, Redactor)


def test_redactor_empty_and_none_mode() -> None:
    """Verifies that empty string or mode='none' returns unmodified text."""
    redactor_none = RegexRedactor(mode="none")
    assert redactor_none.redact("") == ""
    assert redactor_none.redact("unchanged text") == "unchanged text"


def test_redactor_idempotency(redactor: Redactor) -> None:
    """Verifies idempotency: redact(redact(text)) == redact(text)."""
    text = "Error 0xDEADBEEF occurred at 10.0.0.1 on node-5."
    once = redactor.redact(text)
    twice = redactor.redact(once)
    assert once == twice
