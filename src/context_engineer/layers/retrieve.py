"""Layer 3: Bounded memory and context retrieval across dropped turns (Algorithm A9).

Retrieves top-k relevant turns from evicted history using BM25 with strict token budget bounding.
Never mutates the window boundary, summary, or prefix zone.
"""

import dataclasses
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from context_engineer.config import ContextConfig
from context_engineer.protocols import Retriever
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer, truncate_text_to_tokens
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


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text for BM25, splitting identifiers like worker-12 into sub-tokens."""
    tokens: list[str] = []
    # Find all alphanumeric sequences and hyphenated words
    for word in re.findall(r"[\w.-]+", text.lower()):
        tokens.append(word)
        # Split on hyphens/underscores/dots to index sub-components
        subparts = [p for p in re.split(r"[-_.]", word) if p and p != word]
        tokens.extend(subparts)
    return tokens or ["<empty>"]


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


class BM25Retriever(Retriever):
    """Deterministic BM25 retriever bounding hits within allocated token budget (Algorithm A9)."""

    def retrieve(
        self,
        query: str,
        candidates: Sequence[Turn],
        *,
        k: int,
        token_budget: int,
        tok: Tokenizer,
    ) -> list[Turn]:
        """Retrieves top-k relevant turns bounded within the allocated token budget."""
        if not candidates or not query or token_budget <= 0 or k <= 0:
            return []

        # 1. Tokenize corpus
        corpus_tokens = [_tokenize_for_bm25(t.get_uncapped_text()) for t in candidates]
        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)

        # 2. Rank candidates: score desc, then id desc (most recent first)
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda idx: (scores[idx], candidates[idx].id),
            reverse=True,
        )

        selected: list[Turn] = []
        used_tokens = 0

        for idx in ranked_indices:
            if len(selected) >= k:
                break
            score = float(scores[idx])
            if score <= 0.0:
                continue

            candidate = candidates[idx]
            uncapped = get_uncapped_turn(candidate)
            cost = tok.count(uncapped.get_effective_text())

            if used_tokens + cost <= token_budget:
                selected.append(uncapped)
                used_tokens += cost
            else:
                remaining_tokens = token_budget - used_tokens
                if remaining_tokens > 20:
                    # Truncate hit at line/sentence boundary so it strictly fits within budget
                    truncated_content = truncate_text_to_tokens(
                        uncapped.content,
                        max_tokens=remaining_tokens,
                        tokenizer=tok,
                    )
                    bounded_turn = dataclasses.replace(
                        uncapped,
                        content=truncated_content,
                        tool_output=None,
                    )
                    selected.append(bounded_turn)
                    used_tokens += tok.count(bounded_turn.get_effective_text())
                break

        # Return hits in deterministic chronological order
        return sorted(selected, key=lambda t: t.id)


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
