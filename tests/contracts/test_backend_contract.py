"""Contract tests for Backend implementations."""

import pytest

from context_engineer.backends.mock import MockBackend
from context_engineer.protocols import Backend, ChatResult
from context_engineer.types import Message


@pytest.fixture
def mock_backend() -> Backend:
    return MockBackend(default_response="Mock reply.")


def test_backend_protocol_conformance(mock_backend: Backend) -> None:
    """Verifies that the backend conforms to the Backend protocol."""
    assert isinstance(mock_backend, Backend)


@pytest.mark.anyio
async def test_backend_chat_contract(mock_backend: Backend) -> None:
    """Verifies async chat dispatch contract."""
    messages = [
        Message(role="system", content="System instruction."),
        Message(role="user", content="User question."),
    ]
    result = await mock_backend.chat(messages, model="mock-model")

    assert isinstance(result, ChatResult)
    assert result.content == "Mock reply."
    assert result.finish_reason == "stop"
    assert result.cache is not None
    assert result.cache.measured is False


def test_backend_sync_wrapper(mock_backend: Backend) -> None:
    """Verifies synchronous convenience wrapper if available."""
    if hasattr(mock_backend, "chat_sync"):
        messages = [Message(role="user", content="Ping")]
        result = mock_backend.chat_sync(messages, model="mock-model")
        assert isinstance(result, ChatResult)
        assert result.content == "Mock reply."
