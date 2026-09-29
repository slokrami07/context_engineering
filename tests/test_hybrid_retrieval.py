"""Tests verifying hybrid retrieval and zero-change pipeline protocol swapping (Phase 3 Acceptance Criteria)."""

from collections.abc import Sequence

from context_engineer.config import ContextConfig
from context_engineer.pipeline import assemble_context
from context_engineer.protocols import DenseEmbedder
from context_engineer.retrievers.bm25 import BM25Retriever
from context_engineer.retrievers.dense import DenseRetriever
from context_engineer.retrievers.hybrid import HybridRetriever
from context_engineer.summarizers.extractive import ExtractiveSummarizer
from context_engineer.tokenizers.approx import ApproxTokenizer
from context_engineer.types import Turn


class ParaphraseSemanticEmbedder(DenseEmbedder):
    """Semantic embedder mapping semantically equivalent concepts to close vectors."""

    def __init__(self) -> None:
        # Semantic concept mapping
        self.concepts = {
            "concurrency": 0,
            "thread": 0,
            "scheduling": 1,
            "ordered": 1,
            "serializable": 2,
            "race": 2,
            "dirty": 3,
            "conflicts": 3,
            "network": 4,
            "packet": 4,
            "log": 5,
            "credentials": 5,
        }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 6
            for word in text.lower().split():
                clean = word.strip(".,!?:")
                if clean in self.concepts:
                    vec[self.concepts[clean]] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            results.append([x / norm for x in vec] if norm > 0 else vec)
        return results


def test_zero_pipeline_changes_for_swapping_implementations() -> None:
    """Verifies that any Retriever or Summarizer implementation can be swapped with zero pipeline changes."""
    config = ContextConfig(context_ceiling=2048, completion_reserve=256)
    turns = [
        Turn(id=1, role="user", content="Deploy v1.0"),
        Turn(id=2, role="assistant", content="v1.0 deployed"),
    ]

    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRetriever()
    summarizer = ExtractiveSummarizer()

    # Zero pipeline code changes: pass different retrievers directly
    res_bm25 = assemble_context(
        turns, query="status", config=config, retriever=bm25, summarizer=summarizer
    )
    res_dense = assemble_context(
        turns, query="status", config=config, retriever=dense, summarizer=summarizer
    )
    res_hybrid = assemble_context(
        turns, query="status", config=config, retriever=hybrid, summarizer=summarizer
    )

    assert res_bm25.budget_plan.is_valid
    assert res_dense.budget_plan.is_valid
    assert res_hybrid.budget_plan.is_valid


def test_retrieval_hard_bounded() -> None:
    """Verifies that retrieval never returns an uncapped, unbounded turn that exceeds token budget."""
    retriever = HybridRetriever()
    tok = ApproxTokenizer()

    oversized_turn = Turn(
        id=1,
        role="assistant",
        content="datum " * 500,  # ~500 tokens
    )

    budget = 45  # strictly bounded
    hits = retriever.retrieve("datum", [oversized_turn], k=1, token_budget=budget, tok=tok)

    assert len(hits) == 1
    hit_tokens = tok.count(hits[0].get_effective_text())
    assert hit_tokens <= budget


def test_hybrid_retriever_beats_bm25_on_paraphrase_scenario() -> None:
    """Verifies that HybridRetriever beats BM25 on the paraphrase scenario with no rare-token overlap."""
    tok = ApproxTokenizer()
    embedder = ParaphraseSemanticEmbedder()

    dense_retriever = DenseRetriever(embedder=embedder)
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever, dense_retriever=dense_retriever
    )

    # Needle: contains semantic answer about concurrency and scheduling, no token overlap with query
    needle = Turn(
        id=1,
        role="assistant",
        content="The concurrency protocol enforces serializable transaction scheduling preventing dirty reads.",
    )
    # Lexical distractor: shares lexical tokens with query (execution, data) but is semantically unrelated
    distractor_lexical = Turn(
        id=2,
        role="assistant",
        content="Background execution logs recorded output data from batch jobs.",
    )
    # Network distractor
    distractor_network = Turn(
        id=3,
        role="assistant",
        content="Network protocols specify the maximum transmission unit and packet header layout.",
    )

    candidates = [needle, distractor_lexical, distractor_network]

    # Paraphrased query: expresses the concept without exact keyword overlap with needle
    paraphrase_query = "How is thread execution ordered to avoid data race conflicts?"

    # 1. BM25 retrieval
    bm25_hits = bm25_retriever.retrieve(
        paraphrase_query, candidates, k=1, token_budget=200, tok=tok
    )

    # 2. Hybrid retrieval
    hybrid_hits = hybrid_retriever.retrieve(
        paraphrase_query, candidates, k=1, token_budget=200, tok=tok
    )

    # BM25 fails to retrieve the needle because of zero keyword overlap
    bm25_top_id = bm25_hits[0].id if bm25_hits else None
    assert bm25_top_id != needle.id, (
        f"BM25 unexpectedly matched needle with top hit id={bm25_top_id}"
    )

    # Hybrid retriever successfully identifies and retrieves the needle
    assert len(hybrid_hits) == 1
    assert hybrid_hits[0].id == needle.id, (
        f"Hybrid retriever expected needle id={needle.id}, got {hybrid_hits[0].id}"
    )
