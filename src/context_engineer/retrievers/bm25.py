"""BM25 memory and context retrieval with index caching and small-corpus stability (Algorithm A9).

Fixes:
- R4: Caches the BM25 index keyed by sha256(ordered [(turn.id, content_hash)]).
- R5: Floors IDF to avoid zero/negative scores on small corpora of 1-3 documents.
- R6: Optional query expansion over recent turn history.
- R8: Deterministic tie-breaking by score desc, then most recent turn ID desc.
"""

import dataclasses
import hashlib
import re
from collections.abc import Sequence
from typing import Literal

from rank_bm25 import BM25Okapi

from context_engineer.protocols import Retriever
from context_engineer.tokenizers import Tokenizer, truncate_text_to_tokens
from context_engineer.types import Turn

# LRU index cache: corpus_hash -> (bm25_index, candidate_ids)
_BM25_INDEX_CACHE: dict[str, tuple[BM25Okapi, list[int]]] = {}


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text for BM25, splitting compound identifiers into sub-tokens."""
    tokens: list[str] = []
    for word in re.findall(r"[\w.-]+", text.lower()):
        tokens.append(word)
        subparts = [p for p in re.split(r"[-_.]", word) if p and p != word]
        tokens.extend(subparts)
    return tokens or ["<empty>"]


def _get_uncapped_turn(turn: Turn) -> Turn:
    """Restores original uncapped content and tool_output for retrieval."""
    content = turn.raw_content if turn.raw_content is not None else turn.content
    tool_output = turn.raw_tool_output if turn.raw_tool_output is not None else turn.tool_output
    return dataclasses.replace(
        turn,
        content=content,
        tool_output=tool_output,
        is_capped=False,
    )


def _compute_corpus_hash(candidates: Sequence[Turn]) -> str:
    """Computes deterministic hash over candidate turn IDs and content hashes."""
    hasher = hashlib.sha256()
    for t in candidates:
        hasher.update(f"{t.id}:{t.content_hash}".encode())
    return hasher.hexdigest()


class BM25Retriever(Retriever):
    """Deterministic BM25 retriever bounding hits within allocated token budget (Algorithm A9)."""

    def __init__(
        self,
        query_expansion: Literal["none", "last_n_turns"] = "none",
        min_score: float = 0.0,
    ) -> None:
        self.query_expansion = query_expansion
        self.min_score = min_score

    def _get_or_build_index(self, candidates: Sequence[Turn]) -> BM25Okapi:
        """Retrieves cached BM25 index or builds and caches a new one (Fix R4)."""
        corpus_hash = _compute_corpus_hash(candidates)
        if corpus_hash in _BM25_INDEX_CACHE:
            return _BM25_INDEX_CACHE[corpus_hash][0]

        corpus_tokens = [_tokenize_for_bm25(t.get_uncapped_text()) for t in candidates]
        bm25 = BM25Okapi(corpus_tokens)

        # Fix R5: On small corpora (1-3 docs), Okapi formula can produce 0 or negative IDFs.
        # Floor all non-zero IDFs to at least 0.25 to guarantee meaningful scores.
        for word, idf in list(bm25.idf.items()):
            if idf <= 0.0:
                bm25.idf[word] = 0.25

        if len(_BM25_INDEX_CACHE) > 100:
            _BM25_INDEX_CACHE.clear()

        _BM25_INDEX_CACHE[corpus_hash] = (bm25, [t.id for t in candidates])
        return bm25

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

        # Optional query expansion
        effective_query = query
        if self.query_expansion == "last_n_turns" and len(candidates) > 0:
            recent_context = " ".join(t.content for t in candidates[-2:])
            effective_query = f"{query} {recent_context}"

        query_tokens = _tokenize_for_bm25(effective_query)
        if not query_tokens:
            return []

        bm25 = self._get_or_build_index(candidates)
        scores = bm25.get_scores(query_tokens)

        # Fix R8: Rank candidates by score desc, then most recent turn ID desc
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
            if score <= self.min_score:
                continue

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
