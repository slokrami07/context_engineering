"""Honest chronological RAG baseline.

Unlike the legacy strawman which hardcoded needle text and keyword sniffing,
this honest baseline:
1. Allocates a recent conversation window.
2. Uses the shared BM25Retriever to search evicted dropped turns.
3. Inserts retrieved hits at their original chronological index inside the conversation.
4. Uses identical token budgeting and accounting.
"""

from benchmarks.baselines.base import Baseline, BaselineResult
from context_engineer.config import ContextConfig
from context_engineer.retrievers.bm25 import BM25Retriever
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


class ChronologicalRAGBaseline(Baseline):
    """Honest Chronological RAG inserting retrieved memories at their original chronological place."""

    def __init__(self, retriever: BM25Retriever | None = None) -> None:
        self.retriever = retriever or BM25Retriever()

    @property
    def name(self) -> str:
        return "Chronological RAG"

    def assemble(
        self,
        turns: list[Turn],
        query: str,
        system_prompt: str,
        config: ContextConfig,
        tok: Tokenizer,
    ) -> BaselineResult:
        sys_tokens = tok.count(system_prompt)
        query_tokens = tok.count(query)
        retrieval_budget = max(0, config.suffix_reserve_tokens - query_tokens)

        avail_for_window = (
            config.effective_budget - sys_tokens - query_tokens - retrieval_budget - 20
        )
        if avail_for_window <= 0:
            return BaselineResult(
                name=self.name,
                messages=[Message("system", system_prompt), Message("user", query)],
                prompt_tokens=sys_tokens + query_tokens,
                overflow=True,
            )

        packed_turns: list[Turn] = []
        used = 0
        split_idx = len(turns)

        for i, t in enumerate(reversed(turns)):
            cost = tok.count(t.get_effective_text())
            if used + cost <= avail_for_window:
                packed_turns.append(t)
                used += cost
            else:
                split_idx = len(turns) - i
                break

        window_turns = list(reversed(packed_turns))
        dropped_turns = turns[:split_idx]

        # Retrieve relevant memories from dropped history using shared BM25Retriever
        retrieved_hits: list[Turn] = []
        if dropped_turns:
            retrieved_hits = self.retriever.retrieve(
                query=query,
                candidates=dropped_turns,
                k=config.retrieval_top_k,
                token_budget=retrieval_budget,
                tok=tok,
            )

        # In Chronological RAG, retrieved turns are inserted at their chronological place
        combined_turns = sorted(
            window_turns + retrieved_hits,
            key=lambda t: t.id,
        )

        messages = [Message("system", system_prompt)]
        for t in combined_turns:
            messages.append(Message(t.role, t.get_effective_text()))
        messages.append(Message("user", query))

        total_tokens = sum(tok.count(m.content) for m in messages)
        return BaselineResult(
            name=self.name,
            messages=messages,
            prompt_tokens=total_tokens,
            overflow=False,
            retrieved_turns=retrieved_hits,
        )
