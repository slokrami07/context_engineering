"""Layer 2: Pinned turn separation and fixed prefix budgeting (Algorithm A1 and Invariant I8).

Separates invariant pinned facts from conversational history. Pinned turns are never capped,
and their token cost is budgeted as part of the immutable prefix zone.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from context_engineer.config import ContextConfig
from context_engineer.errors import ContextBudgetError
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer
from context_engineer.types import Message, Turn


@dataclass(frozen=True, slots=True)
class PinResult:
    """Output of the Pin Layer."""

    pinned_turns: list[Turn]
    unpinned_turns: list[Turn]
    system_tokens: int
    pinned_tokens: int
    fixed_prefix_tokens: int
    remaining_budget: int = 0

    @property
    def reserved_tokens(self) -> int:
        """Deprecated alias for fixed_prefix_tokens."""
        return self.fixed_prefix_tokens


def split_pinned(turns: Sequence[Turn]) -> tuple[list[Turn], list[Turn]]:
    """Splits conversational history into pinned invariant turns and unpinned turns."""
    pinned: list[Turn] = []
    unpinned: list[Turn] = []
    for t in turns:
        if t.pinned:
            pinned.append(t)
        else:
            unpinned.append(t)
    return pinned, unpinned


def apply_pin_layer(
    turns: Sequence[Turn],
    system_prompt: str,
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> PinResult:
    """Separates pinned turns and calculates invariant prefix token costs (Algorithm A1).

    Args:
        turns: List of conversational turns.
        system_prompt: Base system prompt.
        config: Central configuration instance.
        tokenizer: Tokenizer instance.

    Returns:
        PinResult containing partitioned turns and accurate token counts.

    Raises:
        ContextBudgetError: If fixed prefix cost alone exceeds the effective budget ceiling (I9).
    """
    cfg = config or ContextConfig()
    tok = tokenizer or resolve_tokenizer(cfg.tokenizer)

    pinned, unpinned = split_pinned(turns)

    # 1. System prompt cost
    sys_msg = Message("system", system_prompt)
    system_tokens = tok.count_messages([sys_msg], add_generation_prompt=False)

    # 2. Pinned facts cost
    pinned_tokens = 0
    if pinned:
        pinned_facts = "\n".join(f"- {p.get_effective_text()}" for p in pinned)
        pinned_msg = Message("system", f"Pinned facts:\n{pinned_facts}")
        pinned_tokens = tok.count_messages([pinned_msg], add_generation_prompt=False)

    fixed_prefix_tokens = system_tokens + pinned_tokens

    if fixed_prefix_tokens >= cfg.effective_budget:
        raise ContextBudgetError(
            f"Fixed prefix tokens ({fixed_prefix_tokens}) exceed effective budget "
            f"({cfg.effective_budget}). Reduce pinned facts or system prompt size."
        )

    return PinResult(
        pinned_turns=pinned,
        unpinned_turns=unpinned,
        system_tokens=system_tokens,
        pinned_tokens=pinned_tokens,
        fixed_prefix_tokens=fixed_prefix_tokens,
        remaining_budget=cfg.effective_budget - fixed_prefix_tokens,
    )
