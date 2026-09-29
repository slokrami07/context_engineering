"""Raw append-only conversation baseline."""

from benchmarks.baselines.base import Baseline, BaselineResult
from context_engineer.config import ContextConfig
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


class RawAppendBaseline(Baseline):
    """Raw append-only baseline with zero truncation, tracking context ceiling overflow."""

    @property
    def name(self) -> str:
        return "Raw Append-Only"

    def assemble(
        self,
        turns: list[Turn],
        query: str,
        system_prompt: str,
        config: ContextConfig,
        tok: Tokenizer,
    ) -> BaselineResult:
        messages = [Message("system", system_prompt)]
        for t in turns:
            messages.append(Message(t.role, t.get_effective_text()))
        messages.append(Message("user", query))

        total_tokens = sum(tok.count(m.content) for m in messages)
        overflow = total_tokens > config.effective_budget

        return BaselineResult(
            name=self.name,
            messages=messages,
            prompt_tokens=total_tokens,
            overflow=overflow,
            overflow_turn=turns[-1].id if overflow and turns else None,
        )
