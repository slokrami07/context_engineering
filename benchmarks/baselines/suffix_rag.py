"""Suffix-RAG unquantized baseline (Context-Engineer ablation without Algorithm A2).

Places retrieved memories in the prompt suffix alongside the query,
but advances the recency window boundary on every single turn (chunk_tokens=1).
This directly isolates and measures the value of Algorithm A2 (quantized window boundary).
"""

from benchmarks.baselines.base import Baseline, BaselineResult
from context_engineer.config import ContextConfig
from context_engineer.retrievers.bm25 import BM25Retriever
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message, Turn


class SuffixRAGUnquantizedBaseline(Baseline):
    """Suffix-RAG with per-turn sliding window (Context-Engineer without A2 quantization)."""

    def __init__(self, retriever: BM25Retriever | None = None) -> None:
        self.retriever = retriever or BM25Retriever()

    @property
    def name(self) -> str:
        return "Suffix-RAG (Unquantized Window)"

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

        # Unquantized: greedily slide window by 1 turn
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

        messages = [Message("system", system_prompt)]
        for t in window_turns:
            messages.append(Message(t.role, t.get_effective_text()))

        # Suffix-RAG: append retrieval as suffix block before query
        if retrieved_hits:
            retrieval_text = "\n".join(
                f"[Retrieved memory #{h.id}]: {h.get_effective_text()}" for h in retrieved_hits
            )
            messages.append(Message("system", f"Context memory:\n{retrieval_text}"))

        messages.append(Message("user", query))

        total_tokens = sum(tok.count(m.content) for m in messages)
        return BaselineResult(
            name=self.name,
            messages=messages,
            prompt_tokens=total_tokens,
            overflow=False,
            retrieved_turns=retrieved_hits,
        )
