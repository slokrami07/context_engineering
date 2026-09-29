"""Base protocol and result structures for benchmark baselines."""

import dataclasses
from typing import Protocol

from context_engineer.config import ContextConfig
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


@dataclasses.dataclass(frozen=True, slots=True)
class BaselineResult:
    """Result of assembling messages with a baseline strategy."""

    name: str
    messages: list[Message]
    prompt_tokens: int
    overflow: bool = False
    overflow_turn: int | None = None
    retrieved_turns: list[Turn] = dataclasses.field(default_factory=list)


class Baseline(Protocol):
    """Protocol for a context management strategy baseline."""

    @property
    def name(self) -> str:
        """Name of the baseline strategy."""
        ...

    def assemble(
        self,
        turns: list[Turn],
        query: str,
        system_prompt: str,
        config: ContextConfig,
        tok: Tokenizer,
    ) -> BaselineResult:
        """Assembles prompt messages for the given dialog history and query."""
        ...
