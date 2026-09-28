"""Layer 4: Recency-based contiguous turn packing.

Greedily fits the most recent capped turns into the remaining token balance.
Guarantees contiguous conversational history to maintain conversational flow.
"""

from dataclasses import dataclass

from context_engineer.config import ContextConfig
from context_engineer.tokenizer import count_turn_tokens
from context_engineer.types import Turn


@dataclass
class WindowResult:
    """Output of the Window Layer."""

    window_turns: list[Turn]
    dropped_turns: list[Turn]
    window_tokens: int


def apply_window_layer(
    unpinned_turns: list[Turn],
    available_budget: int,
    config: ContextConfig | None = None,
    excluded_turn_ids: set[int] | None = None,
) -> WindowResult:
    """Greedily fits the most recent contiguous turns into available_budget.

    Args:
        unpinned_turns: List of unpinned turns in chronological order.
        available_budget: Max tokens available for the recency window.
        config: Central configuration.
        excluded_turn_ids: Turn IDs already reserved (e.g., retrieved turn).

    Returns:
        WindowResult with packed contiguous window turns and dropped older turns.
    """
    if config is None:
        config = ContextConfig()

    if excluded_turn_ids is None:
        excluded_turn_ids = set()

    # Filter out turns that are explicitly excluded (e.g. retrieved turn)
    candidate_turns = [t for t in unpinned_turns if t.id not in excluded_turn_ids]

    if not candidate_turns:
        return WindowResult(window_turns=[], dropped_turns=[], window_tokens=0)

    # Greedily pack from the latest turn backwards to guarantee contiguous recency
    packed_turns: list[Turn] = []
    used_tokens = 0
    split_index = len(candidate_turns)

    for i in range(len(candidate_turns) - 1, -1, -1):
        turn = candidate_turns[i]
        cost = count_turn_tokens(turn, model_name=config.model_name)
        if used_tokens + cost <= available_budget:
            packed_turns.append(turn)
            used_tokens += cost
            split_index = i
        else:
            # Cannot fit this turn contiguously without breaking recency sequence
            break

    # packed_turns was collected backwards; reverse to chronological order
    window_turns = list(reversed(packed_turns))
    dropped_turns = candidate_turns[:split_index]

    return WindowResult(
        window_turns=window_turns,
        dropped_turns=dropped_turns,
        window_tokens=used_tokens,
    )
