"""Layer 3: Lookahead rank-bm25 selector.

Determines which turns will be omitted by the recency window given the budget.
Runs BM25 over the dropped turns against the current user query. Pulls the top
matching turn UNCAPPED (restoring raw original content / tool outputs) and
reserves its token budget for dynamic suffix injection.
"""

import dataclasses
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from context_engineer.config import ContextConfig
from context_engineer.layers.window import apply_window_layer
from context_engineer.tokenizer import count_turn_tokens
from context_engineer.types import Turn


@dataclass
class RetrieveResult:
    """Output of the Retrieve Layer."""

    retrieved_turn: Turn | None
    retrieved_turn_id: int | None
    score: float
    retrieval_tokens: int


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text into normalized lowercase alphanumeric tokens for BM25."""
    return [w.lower() for w in re.findall(r"\w+", text) if w]


def get_uncapped_turn(turn: Turn) -> Turn:
    """Returns a copy of the turn with raw uncapped content and tool_output restored."""
    content = turn.raw_content if turn.raw_content is not None else turn.content
    tool_output = turn.raw_tool_output if turn.raw_tool_output is not None else turn.tool_output
    return dataclasses.replace(
        turn,
        content=content,
        tool_output=tool_output,
        is_capped=False,
    )


def apply_retrieve_layer(
    unpinned_turns: list[Turn],
    query: str,
    available_budget: int,
    config: ContextConfig | None = None,
) -> RetrieveResult:
    """Layer 3 pipeline entry point: retrieves relevant dropped turns via BM25.

    Args:
        unpinned_turns: List of unpinned turns (Layer 1 capped).
        query: Current user question / query string.
        available_budget: Approximate remaining token budget available before retrieval.
        config: Central configuration.

    Returns:
        RetrieveResult containing the uncapped retrieved turn and token cost.
    """
    if config is None:
        config = ContextConfig()

    if not unpinned_turns or not query:
        return RetrieveResult(None, None, 0.0, 0)

    # 1. Lookahead simulation: Determine which turns would be omitted by the window
    simulated_window = apply_window_layer(
        unpinned_turns=unpinned_turns,
        available_budget=available_budget,
        config=config,
    )
    dropped_candidates = simulated_window.dropped_turns

    if not dropped_candidates:
        # All turns fit inside the contiguous recency window; no need to retrieve dropped turns
        return RetrieveResult(None, None, 0.0, 0)

    # 2. Tokenize corpus of dropped turns for BM25
    corpus_tokens: list[list[str]] = []
    for turn in dropped_candidates:
        # Use full uncapped text for BM25 matching to ensure complete signal match
        uncapped_text = turn.get_uncapped_text()
        tokens = _tokenize_for_bm25(uncapped_text)
        if not tokens:
            tokens = ["<empty>"]
        corpus_tokens.append(tokens)

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return RetrieveResult(None, None, 0.0, 0)

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)

    # 3. Find top scoring candidate
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score < config.bm25_min_score:
        return RetrieveResult(None, None, best_score, 0)

    selected_turn = dropped_candidates[best_idx]
    # Re-inject top hit UNCAPPED
    uncapped_turn = get_uncapped_turn(selected_turn)
    uncapped_cost = count_turn_tokens(uncapped_turn, model_name=config.model_name)

    return RetrieveResult(
        retrieved_turn=uncapped_turn,
        retrieved_turn_id=selected_turn.id,
        score=best_score,
        retrieval_tokens=uncapped_cost,
    )
