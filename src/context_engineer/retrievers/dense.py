"""Dense semantic vector retrieval implementing the Retriever protocol (Algorithm A9).

Provides semantic vector retrieval using configurable dense embedders.
Includes a deterministic offline semantic projection embedder as standard fallback
requiring no external network downloads.
"""

import dataclasses
import hashlib
import re
from collections.abc import Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from context_engineer.protocols import DenseEmbedder, Retriever
from context_engineer.tokenizers import Tokenizer, truncate_text_to_tokens
from context_engineer.types import Turn


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


class OfflineSemanticEmbedder(DenseEmbedder):
    """Deterministic offline semantic projection embedder.

    Produces dense embeddings via hashed character-ngram and subword bag projections,
    capturing morphological and subword semantic similarity without downloading weights.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _text_to_vector(self, text: str) -> list[float]:
        if np is None:
            # Fallback simple python float list if numpy not present
            vec = [0.0] * self.dim
            words = re.findall(r"\w+", text.lower())
            if not words:
                return vec
            for word in words:
                # Project word and character 3-grams
                tokens = [word] + [word[i : i + 3] for i in range(len(word) - 2)]
                for t in tokens:
                    h = int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim
                    vec[h] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            return [x / norm for x in vec] if norm > 0 else vec

        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return [float(x) for x in vec]

        for word in words:
            tokens = [word] + [word[i : i + 3] for i in range(len(word) - 2)]
            for t in tokens:
                h = int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim
                vec[h] += 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds a batch of texts into normalized dense vectors."""
        return [self._text_to_vector(t) for t in texts]


class DenseRetriever(Retriever):
    """Dense retriever scoring candidate turns by vector cosine similarity."""

    def __init__(
        self,
        embedder: DenseEmbedder | None = None,
        min_similarity: float = 0.0,
    ) -> None:
        self.embedder = embedder or OfflineSemanticEmbedder()
        self.min_similarity = min_similarity

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

        # 1. Embed query and candidates
        candidate_texts = [t.get_uncapped_text() for t in candidates]
        query_vec = self.embedder.embed([query])[0]
        candidate_vecs = self.embedder.embed(candidate_texts)

        # 2. Compute cosine similarity scores
        scores: list[float] = []
        for c_vec in candidate_vecs:
            # Vectors are L2 normalized
            sim = sum(q * c for q, c in zip(query_vec, c_vec, strict=False))
            scores.append(float(sim))

        # 3. Rank by score desc, then most recent turn ID desc
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
            score = scores[idx]
            if score <= self.min_similarity:
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
