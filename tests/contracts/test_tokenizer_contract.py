"""Contract tests for all Tokenizer implementations."""

import pytest

from context_engineer.protocols import Tokenizer
from context_engineer.tokenizers.approx import ApproxTokenizer
from context_engineer.types import Message


@pytest.fixture(params=["approx"])
def tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    if request.param == "approx":
        return ApproxTokenizer()
    pytest.skip(f"Unsupported tokenizer spec {request.param}")


def test_tokenizer_protocol_conformance(tokenizer: Tokenizer) -> None:
    """Verifies that the tokenizer conforms to the Tokenizer protocol."""
    assert isinstance(tokenizer, Tokenizer)
    assert hasattr(tokenizer, "id")
    assert isinstance(tokenizer.id, str)


def test_tokenizer_counting_contract(tokenizer: Tokenizer) -> None:
    """Verifies counting behavior on empty and non-empty strings."""
    assert tokenizer.count("") == 0
    assert tokenizer.count("Hello, world!") > 0


def test_tokenizer_message_counting_contract(tokenizer: Tokenizer) -> None:
    """Verifies message counting contract including template overhead."""
    msgs = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
    ]
    count_without = tokenizer.count_messages(msgs, add_generation_prompt=False)
    count_with = tokenizer.count_messages(msgs, add_generation_prompt=True)

    assert count_without > 0
    assert count_with >= count_without


def test_tokenizer_encoding_decoding_contract(tokenizer: Tokenizer) -> None:
    """Verifies encode and decode round-trip contract."""
    sample = "Deterministic test string for round-trip encoding."
    tokens = tokenizer.encode(sample)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    # For ApproxTokenizer, count must equal length of encoded ids
    if isinstance(tokenizer, ApproxTokenizer):
        assert len(tokens) == tokenizer.count(sample)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
