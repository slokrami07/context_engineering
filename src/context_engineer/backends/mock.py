"""Mock inference backend for contract testing, offline verification, and benchmarks."""

from collections.abc import Callable, Sequence
from typing import Any

from context_engineer.protocols import Backend, CacheStats, ChatResult
from context_engineer.types import Message


class MockBackend(Backend):
    """Deterministic offline mock backend conforming to Backend protocol."""

    def __init__(
        self,
        default_response: str = "Mock assistant response.",
        handler: Callable[[Sequence[Message], str], str] | None = None,
        simulated_cached_tokens: int = 0,
    ) -> None:
        self.default_response = default_response
        self.handler = handler
        self.simulated_cached_tokens = simulated_cached_tokens
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        stream: bool = False,
        **kw: Any,
    ) -> ChatResult:
        """Dispatches chat completion request to mock runtime."""
        self.calls.append({"messages": list(messages), "model": model, "kw": kw})

        content = self.handler(messages, model) if self.handler else self.default_response

        total_prompt_tokens = sum(len(m.content.split()) for m in messages)
        cache_stats = CacheStats(
            measured=False,
            cached_tokens=self.simulated_cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            cache_hit_rate=(
                self.simulated_cached_tokens / total_prompt_tokens
                if total_prompt_tokens > 0
                else 0.0
            ),
        )

        return ChatResult(
            content=content,
            finish_reason="stop",
            cache=cache_stats,
            raw_response={"mock": True, "model": model},
        )

    def chat_sync(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        stream: bool = False,
        **kw: Any,
    ) -> ChatResult:
        """Synchronous convenience wrapper for non-async callers."""
        import asyncio

        return asyncio.run(self.chat(messages, model=model, stream=stream, **kw))
