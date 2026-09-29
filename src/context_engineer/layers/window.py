"""Layer 4: Stateless quantized window boundary with legal boundary enforcement (Algorithms A2 and A4).

Guarantees prefix stability across consecutive turns by quantizing the window boundary onto a coarse grid.
Prevents orphaning tool results or splitting tool calls from their results.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from context_engineer.config import ContextConfig
from context_engineer.errors import ContextBudgetError
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer
from context_engineer.types import Message, Turn

# Module-level cost memoization: (tokenizer_id, content_hash) -> token count
_TURN_COST_CACHE: dict[tuple[str, str], int] = {}


def memoized_turn_cost(turn: Turn, tok: Tokenizer) -> int:
    """Computes and memoizes the token cost of a turn including message template overhead."""
    cache_key = (tok.id, turn.content_hash)
    if cache_key in _TURN_COST_CACHE:
        return _TURN_COST_CACHE[cache_key]

    msg = Message(role=turn.role, content=turn.get_effective_text())
    cost = tok.count_messages([msg], add_generation_prompt=False)
    # Bound cache size to prevent memory leaks in long-running processes
    if len(_TURN_COST_CACHE) > 10000:
        _TURN_COST_CACHE.clear()
    _TURN_COST_CACHE[cache_key] = cost
    return cost


def is_legal_boundary(
    turns: Sequence[Turn],
    i: int,
    *,
    require_user_start: bool = True,
) -> bool:
    """Evaluates whether index i is a structurally legal window boundary (Algorithm A4).

    Rules:
        1. Boundary at len(turns) is always legal (empty window).
        2. Window turn cannot be a tool_result (would orphan result from call).
        3. Preceding dropped turn cannot be a tool_call (would split call from result).
        4. If require_user_start is True, window must start on a user message turn.
    """
    n = len(turns)
    if i == n:
        return True
    if i < 0 or i > n:
        return False

    t = turns[i]
    if require_user_start and not (t.role == "user" and t.kind == "message"):
        return False

    if t.kind == "tool_result":
        return False

    return not (i > 0 and turns[i - 1].kind == "tool_call")


def next_legal_boundary(
    turns: Sequence[Turn],
    i: int,
    *,
    require_user_start: bool = True,
) -> int:
    """Finds smallest legal boundary index j >= i, falling back to len(turns)."""
    n = len(turns)
    for j in range(max(0, i), n + 1):
        if is_legal_boundary(turns, j, require_user_start=require_user_start):
            return j
    # Fallback without require_user_start if user turn not available
    if require_user_start:
        for j in range(max(0, i), n + 1):
            if is_legal_boundary(turns, j, require_user_start=False):
                return j
    return n


def prev_legal_boundary(
    turns: Sequence[Turn],
    i: int,
    *,
    require_user_start: bool = True,
) -> int:
    """Finds largest legal boundary index j <= i, falling back to 0."""
    n = len(turns)
    start = min(n, max(0, i))
    for j in range(start, -1, -1):
        if is_legal_boundary(turns, j, require_user_start=require_user_start):
            return j
    # Fallback without require_user_start
    if require_user_start:
        for j in range(start, -1, -1):
            if is_legal_boundary(turns, j, require_user_start=False):
                return j
    return 0


def window_boundary(
    turns: Sequence[Turn],
    window_budget: int,
    *,
    chunk_tokens: int | None = None,
    min_recent_turns: int = 4,
    cost_fn: Callable[[Turn], int],
    require_user_start: bool = True,
) -> int:
    """Calculates stateless quantized window boundary (Algorithm A2).

    Quantizes the dropped mass onto chunk_tokens multiples to ensure hysteresis
    and maintain high prefix stability without stored state.
    """
    n = len(turns)
    if n == 0:
        return 0

    # Prefix sums P[i] = sum of costs of turns[:i], P[0] = 0
    costs = [cost_fn(t) for t in turns]
    P = [0] * (n + 1)
    for idx, c in enumerate(costs):
        P[idx + 1] = P[idx] + c

    # Find minimal drop index m where turns[m:] fits within window_budget
    m = 0
    while m < n and (P[n] - P[m]) > window_budget:
        m += 1

    if m == 0:
        if is_legal_boundary(turns, 0, require_user_start=require_user_start):
            return 0
        leg = next_legal_boundary(turns, 0, require_user_start=require_user_start)
        if leg < n and (P[n] - P[leg]) <= window_budget:
            return leg
        return 0

    chunk = chunk_tokens if chunk_tokens and chunk_tokens > 0 else max(1, int(0.20 * window_budget))
    # Quantize dropped mass UP to nearest chunk boundary
    target = math.ceil(P[m] / chunk) * chunk

    # Find smallest i in [m..n] with P[i] >= target
    mq = m
    while mq < n and P[mq] < target:
        mq += 1

    mq = next_legal_boundary(turns, mq, require_user_start=require_user_start)

    # Check min_recent_turns requirement
    if mq > n - min_recent_turns:
        fallback = prev_legal_boundary(
            turns,
            max(m, n - min_recent_turns),
            require_user_start=require_user_start,
        )
        if fallback >= m:
            mq = fallback
        elif is_legal_boundary(turns, m, require_user_start=False):
            mq = m

    if (P[n] - P[mq]) > window_budget:
        raise ContextBudgetError(
            f"Latest turns exceed window budget: {P[n] - P[mq]} tokens needed, "
            f"window budget is {window_budget} tokens."
        )

    return mq


@dataclass(frozen=True, slots=True)
class WindowResult:
    """Output of the Window Layer."""

    window_turns: list[Turn]
    dropped_turns: list[Turn]
    window_tokens: int


def apply_window_layer(
    unpinned_turns: Sequence[Turn],
    available_budget: int,
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> WindowResult:
    """Layer 4 entry point: calculates quantized window turns and dropped turns."""
    cfg = config or ContextConfig()
    tok = tokenizer or resolve_tokenizer(cfg.tokenizer)

    def cost(t: Turn) -> int:
        return memoized_turn_cost(t, tok)

    boundary = window_boundary(
        unpinned_turns,
        available_budget,
        chunk_tokens=cfg.window_chunk_tokens,
        min_recent_turns=cfg.min_recent_turns,
        cost_fn=cost,
        require_user_start=cfg.require_user_start,
    )

    window = list(unpinned_turns[boundary:])
    dropped = list(unpinned_turns[:boundary])
    window_tokens = sum(cost(t) for t in window)

    return WindowResult(
        window_turns=window,
        dropped_turns=dropped,
        window_tokens=window_tokens,
    )
