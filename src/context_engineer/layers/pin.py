"""Layer 2: Fact keeper & system prompt reserve.

Deducts system prompt tokens + invariant facts (pinned turns) from the budget
ceiling, establishing the invariant reserved prefix block.
"""

from dataclasses import dataclass

from context_engineer.config import ContextConfig
from context_engineer.tokenizer import count_message_tokens, count_turn_tokens
from context_engineer.types import Turn


@dataclass
class PinResult:
    """Output of the Pin Layer."""

    pinned_turns: list[Turn]
    unpinned_turns: list[Turn]
    system_tokens: int
    pinned_tokens: int
    reserved_tokens: int
    remaining_budget: int


def apply_pin_layer(
    turns: list[Turn],
    system_prompt: str,
    config: ContextConfig | None = None,
) -> PinResult:
    """Layer 2 pipeline entry point: separates pinned turns and deducts invariant token costs.

    Args:
        turns: List of capped conversational turns from Layer 1.
        system_prompt: System prompt string.
        config: Central configuration instance.

    Returns:
        PinResult containing pinned turns, unpinned turns, and remaining token budget.
    """
    if config is None:
        config = ContextConfig()

    system_tokens = count_message_tokens("system", system_prompt, model_name=config.model_name)

    pinned_turns: list[Turn] = []
    unpinned_turns: list[Turn] = []
    pinned_tokens = 0

    for turn in turns:
        if turn.pinned:
            pinned_turns.append(turn)
            pinned_tokens += count_turn_tokens(turn, model_name=config.model_name)
        else:
            unpinned_turns.append(turn)

    reserved_tokens = system_tokens + pinned_tokens
    remaining_budget = max(0, config.effective_budget - reserved_tokens)

    return PinResult(
        pinned_turns=pinned_turns,
        unpinned_turns=unpinned_turns,
        system_tokens=system_tokens,
        pinned_tokens=pinned_tokens,
        reserved_tokens=reserved_tokens,
        remaining_budget=remaining_budget,
    )
