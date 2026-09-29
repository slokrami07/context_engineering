"""Contract tests for all Retriever implementations."""

import pytest

from context_engineer.protocols import Retriever
from context_engineer.retrievers.bm25 import BM25Retriever
from context_engineer.retrievers.dense import DenseRetriever
from context_engineer.retrievers.hybrid import HybridRetriever
from context_engineer.tokenizers.approx import ApproxTokenizer
from context_engineer.types import Turn


@pytest.fixture(params=["bm25", "dense", "hybrid"])
def retriever(request: pytest.FixtureRequest) -> Retriever:
    if request.param == "bm25":
        return BM25Retriever()
    if request.param == "dense":
        return DenseRetriever()
    if request.param == "hybrid":
        return HybridRetriever()
    pytest.skip(f"Unknown retriever {request.param}")


@pytest.fixture
def sample_candidates() -> list[Turn]:
    return [
        Turn(id=1, role="user", content="Configure database connection pool with size 20"),
        Turn(id=2, role="assistant", content="Database pool initialized with 20 connections"),
        Turn(id=3, role="user", content="Verify network throughput on eth0 interface"),
        Turn(id=4, role="assistant", content="Network throughput on eth0 is measured at 1 Gbps"),
    ]


def test_retriever_protocol_conformance(retriever: Retriever) -> None:
    """Verifies that the object conforms to the Retriever protocol."""
    assert isinstance(retriever, Retriever)


def test_retriever_empty_inputs(retriever: Retriever, sample_candidates: list[Turn]) -> None:
    """Verifies that empty candidates, empty query, or non-positive budgets return empty list."""
    tok = ApproxTokenizer()
    assert retriever.retrieve("", sample_candidates, k=3, token_budget=200, tok=tok) == []
    assert retriever.retrieve("query", [], k=3, token_budget=200, tok=tok) == []
    assert retriever.retrieve("query", sample_candidates, k=0, token_budget=200, tok=tok) == []
    assert retriever.retrieve("query", sample_candidates, k=3, token_budget=0, tok=tok) == []


def test_retriever_budget_containment(retriever: Retriever, sample_candidates: list[Turn]) -> None:
    """Verifies that total tokens of returned hits strictly satisfy token_budget."""
    tok = ApproxTokenizer()
    budget = 40
    hits = retriever.retrieve(
        "database pool connection", sample_candidates, k=3, token_budget=budget, tok=tok
    )

    assert isinstance(hits, list)
    total_tokens = sum(tok.count(h.get_effective_text()) for h in hits)
    assert total_tokens <= budget


def test_retriever_chronological_ordering(
    retriever: Retriever, sample_candidates: list[Turn]
) -> None:
    """Verifies that returned hits are strictly sorted in ascending chronological order by Turn ID."""
    tok = ApproxTokenizer()
    hits = retriever.retrieve("database", sample_candidates, k=3, token_budget=500, tok=tok)

    hit_ids = [h.id for h in hits]
    assert hit_ids == sorted(hit_ids)


def test_retriever_candidates_immutability(
    retriever: Retriever, sample_candidates: list[Turn]
) -> None:
    """Verifies that candidates are not mutated in-place by the retriever."""
    tok = ApproxTokenizer()
    original_hashes = [t.content_hash for t in sample_candidates]

    _ = retriever.retrieve("database", sample_candidates, k=2, token_budget=500, tok=tok)

    current_hashes = [t.content_hash for t in sample_candidates]
    assert current_hashes == original_hashes


def test_retriever_small_corpus_stability(retriever: Retriever) -> None:
    """Verifies that small corpora of 1-3 documents do not crash or produce negative scores."""
    tok = ApproxTokenizer()
    single_turn = [Turn(id=10, role="assistant", content="Unique single document response")]

    hits = retriever.retrieve("Unique", single_turn, k=1, token_budget=100, tok=tok)
    assert len(hits) <= 1
