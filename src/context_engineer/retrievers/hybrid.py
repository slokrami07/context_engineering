"""Hybrid retrieval fusing BM25 lexical and dense semantic ranking via Reciprocal Rank Fusion (Algorithm A9).

Combines exact keyword matching with semantic vector projection:
score(d) = sum(w_m / (rrf_k + rank_m(d))) for m in {bm25, dense}
"""

import dataclasses
from collections.abc import Sequence

from context_engineer.protocols import Retriever
from context_engineer.retrievers.bm25 import BM25Retriever, _get_uncapped_turn, _tokenize_for_bm25
from context_engineer.retrievers.dense import DenseRetriever
from context_engineer.tokenizers import Tokenizer, truncate_text_to_tokens
from context_engineer.types import Turn


class HybridRetriever(Retriever):
    """Hybrid retriever combining lexical BM25 and dense semantic ranking using RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.bm25 = bm25_retriever or BM25Retriever()
        self.dense = dense_retriever or DenseRetriever()
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

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

        n = len(candidates)
        if n == 1:
            hit = _get_uncapped_turn(candidates[0])
            cost = tok.count(hit.get_effective_text())
            if cost <= token_budget:
                return [hit]
            if token_budget > 20:
                truncated = truncate_text_to_tokens(
                    hit.content, max_tokens=token_budget, tokenizer=tok
                )
                return [dataclasses.replace(hit, content=truncated, tool_output=None)]
            return []

        # 1. BM25 scoring & ranking
        query_tokens = _tokenize_for_bm25(query)
        bm25_index = self.bm25._get_or_build_index(candidates)
        bm25_scores = bm25_index.get_scores(query_tokens) if query_tokens else [0.0] * n

        bm25_ranked = sorted(
            range(n),
            key=lambda idx: (bm25_scores[idx], candidates[idx].id),
            reverse=True,
        )
        bm25_ranks = {candidate_idx: rank + 1 for rank, candidate_idx in enumerate(bm25_ranked)}

        # 2. Dense scoring & ranking
        candidate_texts = [t.get_uncapped_text() for t in candidates]
        query_vec = self.dense.embedder.embed([query])[0]
        candidate_vecs = self.dense.embedder.embed(candidate_texts)

        dense_scores = [
            sum(q * c for q, c in zip(query_vec, c_vec, strict=False)) for c_vec in candidate_vecs
        ]
        dense_ranked = sorted(
            range(n),
            key=lambda idx: (dense_scores[idx], candidates[idx].id),
            reverse=True,
        )
        dense_ranks = {candidate_idx: rank + 1 for rank, candidate_idx in enumerate(dense_ranked)}

        # 3. Reciprocal Rank Fusion (RRF)
        # In standard RRF, only channels that retrieved the document (score > 0) contribute.
        rrf_scores: list[tuple[float, float, float, int]] = []
        for idx in range(n):
            score = 0.0
            if bm25_scores[idx] > 0.0:
                rank_bm25 = bm25_ranks.get(idx, n + 1)
                score += self.bm25_weight / (self.rrf_k + rank_bm25)
            if dense_scores[idx] > 0.0:
                rank_dense = dense_ranks.get(idx, n + 1)
                score += self.dense_weight / (self.rrf_k + rank_dense)

            rrf_scores.append((score, dense_scores[idx], bm25_scores[idx], idx))

        # Rank candidates by fused RRF score desc, then dense score, bm25 score, then ID desc
        ranked_by_rrf = sorted(
            rrf_scores,
            key=lambda item: (item[0], item[1], item[2], candidates[item[3]].id),
            reverse=True,
        )

        selected: list[Turn] = []
        used_tokens = 0

        for _score, _d_score, _b_score, idx in ranked_by_rrf:
            if len(selected) >= k:
                break

            candidate = candidates[idx]
            uncapped = _get_uncapped_turn(candidate)
            cost = tok.count(uncapped.get_effective_text())

            if used_tokens + cost <= token_budget:
                selected.append(uncapped)
                used_tokens += cost
            else:
                remaining_tokens = token_budget - used_tokens
                if remaining_tokens > 20:
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

        # Chronological order
        return sorted(selected, key=lambda t: t.id)
