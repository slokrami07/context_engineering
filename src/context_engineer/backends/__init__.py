"""Inference backends and runtime adapters."""

import asyncio
import concurrent.futures
import inspect
from collections.abc import Sequence
from typing import Any, cast

from context_engineer.backends.mock import MockBackend
from context_engineer.protocols import Backend, CacheStats, ChatResult
from context_engineer.types import Message


def run_chat_sync(
    backend: Backend,
    messages: Sequence[Message],
    *,
    model: str,
    stream: bool = False,
    **kw: Any,
) -> ChatResult:
    """Synchronous execution wrapper for async Backend.chat methods."""
    res = backend.chat(messages, model=model, stream=stream, **kw)
    if inspect.iscoroutine(res):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return cast(ChatResult, pool.submit(asyncio.run, res).result())
        return cast(ChatResult, asyncio.run(res))
    return cast(ChatResult, res)


__all__ = ["Backend", "CacheStats", "ChatResult", "MockBackend", "run_chat_sync"]
