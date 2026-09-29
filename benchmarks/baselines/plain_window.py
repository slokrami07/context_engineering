"""Plain left-truncation sliding window baseline.

Packs system prompt, user query, and as many recent turns as fit in the budget.
Older turns beyond the window are permanently discarded with zero retrieval.
"""

from benchmarks.baselines.base import Baseline, BaselineResult
from context_engineer.config import ContextConfig
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


class PlainWindowBaseline(Baseline):
    """Plain left-truncation sliding window baseline."""

    @property
    def name(self) -> str:
        return "Plain Left-Truncation"

    def assemble(
        self,
        turns: list[Turn],
        query: str,
        system_prompt: str,
        config: ContextConfig,
        tok: Tokenizer,
    ) -> BaselineResult:
        avail = config.effective_budget - tok.count(system_prompt) - tok.count(query) - 10
        if avail <= 0:
            return BaselineResult(
                name=self.name,
                messages=[Message("system", system_prompt), Message("user", query)],
                prompt_tokens=tok.count(system_prompt) + tok.count(query),
                overflow=True,
            )

        packed_turns: list[Turn] = []
        used = 0
        for t in reversed(turns):
            cost = tok.count(t.get_effective_text())
            if used + cost <= avail:
                packed_turns.append(t)
                used += cost
            else:
                break

        window_turns = list(reversed(packed_turns))
        messages = [Message("system", system_prompt)]
        for t in window_turns:
            messages.append(Message(t.role, t.get_effective_text()))
        messages.append(Message("user", query))

        total_tokens = sum(tok.count(m.content) for m in messages)
        return BaselineResult(
            name=self.name,
            messages=messages,
            prompt_tokens=total_tokens,
            overflow=False,
        )
