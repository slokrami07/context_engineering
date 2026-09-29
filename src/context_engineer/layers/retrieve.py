"""Layer 3: Bounded memory and context retrieval across dropped turns (Algorithm A9).

Retrieves top-k relevant turns from evicted history using BM25 with strict token budget bounding.
Never mutates the window boundary, summary, or prefix zone.
"""

import dataclasses
from collections.abc import Sequence

from context_engineer.config import ContextConfig
from context_engineer.retrievers.bm25 import (
    BM25Retriever,
)
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer
from context_engineer.types import Turn


@dataclasses.dataclass(frozen=True, slots=True)
class RetrieveResult:
    """Output of the Retrieve Layer."""

    retrieved_turns: list[Turn]
    retrieved_turn_ids: list[int]
    retrieval_tokens: int

    @property
    def retrieved_turn(self) -> Turn | None:
        """Deprecated shim property for backward compatibility with v0.1.0 callers."""
        return self.retrieved_turns[0] if self.retrieved_turns else None

    @property
    def retrieved_turn_id(self) -> int | None:
        """Deprecated shim property for backward compatibility with v0.1.0 callers."""
        return self.retrieved_turn_ids[0] if self.retrieved_turn_ids else None

    @property
    def score(self) -> float:
        """Deprecated score property."""
        return 1.0 if self.retrieved_turns else 0.0


def apply_retrieve_layer(
    dropped_candidates: Sequence[Turn] | None = None,
    query: str = "",
    token_budget: int = 0,
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
    *,
    unpinned_turns: Sequence[Turn] | None = None,
    available_budget: int | None = None,
) -> RetrieveResult:
    """Layer 3 pipeline entry point: retrieves relevant dropped turns within allocated token budget.

    Args:
        dropped_candidates: List of older turns evicted from the recency window.
        query: Current user question / query string.
        token_budget: Strictly bounded token allowance for retrieved hits.
        config: Central configuration instance.
        tokenizer: Tokenizer instance.
        unpinned_turns: Legacy parameter: unpinned turns to simulate window on.
        available_budget: Legacy parameter: available token budget.

    Returns:
        RetrieveResult containing bounded chronological hits.
    """
    cfg = config or ContextConfig()
    tok = tokenizer or resolve_tokenizer(cfg.tokenizer)

    candidates = dropped_candidates
    budget = token_budget

    if candidates is None and unpinned_turns is not None:
        from context_engineer.layers.window import apply_window_layer

        simulated_budget = available_budget or 100
        win_res = apply_window_layer(
            unpinned_turns,
            available_budget=simulated_budget,
            config=cfg,
            tokenizer=tok,
        )
        candidates = win_res.dropped_turns
        if budget <= 0:
            budget = simulated_budget

    if candidates is None:
        candidates = []

    retriever = BM25Retriever()
    hits = retriever.retrieve(
        query=query,
        candidates=candidates,
        k=cfg.retrieval_top_k,
        token_budget=budget,
        tok=tok,
    )

    used_tokens = sum(tok.count(h.get_effective_text()) for h in hits)
    hit_ids = [h.id for h in hits]

    return RetrieveResult(
        retrieved_turns=hits,
        retrieved_turn_ids=hit_ids,
        retrieval_tokens=used_tokens,
    )
