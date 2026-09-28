"""Prefix-cache hit/miss simulator and token cost ledger.

Simulates KV-cache prefix matching behavior across consecutive inference queries
for local runtimes (such as LM Studio / llama.cpp / vLLM). Compares:
  (a) Raw append-only history (overflows ceiling)
  (b) Naive sliding window with chronological retrieval (cache thrashing)
  (c) Context-Engineer 5-stage pipeline (prefix-stable ribbon with dynamic suffix)
"""

from dataclasses import dataclass, field

from context_engineer.config import ContextConfig
from context_engineer.pipeline import assemble_context
from context_engineer.tokenizer import count_tokens, get_tokenizer
from context_engineer.types import Message, Turn


@dataclass
class StrategyMetrics:
    """Cumulative metrics for a context management strategy."""

    strategy_name: str
    total_prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    computed_tokens: int = 0
    overflow_occurred: bool = False
    overflow_turn: int | None = None
    step_records: list[dict] = field(default_factory=list)

    @property
    def cache_hit_ratio(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return (self.cache_hit_tokens / self.total_prompt_tokens) * 100.0


def _calculate_common_prefix(prev_tokens: list[int], curr_tokens: list[int]) -> int:
    """Calculates the length of the longest identical prefix starting at index 0."""
    match_len = 0
    min_len = min(len(prev_tokens), len(curr_tokens))
    while match_len < min_len and prev_tokens[match_len] == curr_tokens[match_len]:
        match_len += 1
    return match_len


class PrefixCacheLedger:
    """Tracks and simulates prefix-cache performance across multiple turns and queries."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()
        self.tokenizer = get_tokenizer(self.config.model_name)

    def _tokens_for_messages(self, messages: list[Message]) -> list[int]:
        """Converts messages to a single deterministic token stream for prefix comparison."""
        full_text = ""
        for m in messages:
            full_text += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
        full_text += "<|im_start|>assistant\n"
        return self.tokenizer.encode(full_text)

    def simulate(
        self,
        turns: list[Turn],
        queries: list[tuple[int, str]],
        system_prompt: str,
    ) -> dict[str, StrategyMetrics]:
        """Runs multi-query simulation across strategies.

        Args:
            turns: Full conversation history.
            queries: List of (turn_depth, query_text).
            system_prompt: Baseline system instructions.

        Returns:
            Dictionary mapping strategy name to StrategyMetrics.
        """
        metrics = {
            "Raw Append-Only": StrategyMetrics("Raw Append-Only"),
            "Chronological RAG Window": StrategyMetrics("Chronological RAG Window"),
            "Context-Engineer": StrategyMetrics("Context-Engineer"),
        }

        prev_tokens = {
            "Raw Append-Only": [],
            "Chronological RAG Window": [],
            "Context-Engineer": [],
        }

        for turn_depth, query_text in queries:
            active_turns = turns[:turn_depth]

            # -------------------------------------------------------------
            # Strategy A: Raw Append-Only
            # -------------------------------------------------------------
            raw_strat = metrics["Raw Append-Only"]
            if not raw_strat.overflow_occurred:
                raw_msgs = [Message("system", system_prompt)]
                for t in active_turns:
                    raw_msgs.append(Message(t.role, t.get_effective_text()))
                raw_msgs.append(Message("user", query_text))

                tokens = self._tokens_for_messages(raw_msgs)
                tok_len = len(tokens)

                if tok_len > self.config.effective_budget:
                    raw_strat.overflow_occurred = True
                    raw_strat.overflow_turn = turn_depth
                else:
                    hit = _calculate_common_prefix(prev_tokens["Raw Append-Only"], tokens)
                    miss = tok_len - hit
                    raw_strat.total_prompt_tokens += tok_len
                    raw_strat.cache_hit_tokens += hit
                    raw_strat.computed_tokens += miss
                    prev_tokens["Raw Append-Only"] = tokens
                    raw_strat.step_records.append(
                        {
                            "depth": turn_depth,
                            "tokens": tok_len,
                            "hit": hit,
                            "miss": miss,
                        }
                    )

            # -------------------------------------------------------------
            # Strategy B: Chronological RAG Window
            # (Inserts retrieved memories chronologically inside history; evicts left turns)
            # -------------------------------------------------------------
            chrono_strat = metrics["Chronological RAG Window"]
            chrono_msgs = [Message("system", system_prompt)]

            # In Chronological RAG, older turns are evicted when budget is exceeded.
            # If a memory is retrieved (e.g. from turn 14), it is inserted chronologically
            # near turn 14, which shifts and invalidates every token after it!
            avail = (
                self.config.effective_budget
                - count_tokens(system_prompt)
                - count_tokens(query_text)
                - 50
            )
            packed_turns = []
            used = 0
            for t in reversed(active_turns):
                cost = count_tokens(t.get_effective_text()) + 10
                if used + cost <= avail:
                    packed_turns.append(t)
                    used += cost
                else:
                    break

            # If the needle is not in packed_turns and query asks about partition, inject at top of history
            chrono_turn_list = list(reversed(packed_turns))
            if "partition" in query_text.lower() or "blame" in query_text.lower():
                # Injected at the head of conversational turns (chronological)
                chrono_msgs.append(
                    Message(
                        "system", "[Retrieved earlier]: shard-19 suffered write queue deadlock."
                    )
                )

            for t in chrono_turn_list:
                chrono_msgs.append(Message(t.role, t.get_effective_text()))
            chrono_msgs.append(Message("user", query_text))

            tokens = self._tokens_for_messages(chrono_msgs)
            tok_len = len(tokens)
            hit = _calculate_common_prefix(prev_tokens["Chronological RAG Window"], tokens)
            miss = tok_len - hit
            chrono_strat.total_prompt_tokens += tok_len
            chrono_strat.cache_hit_tokens += hit
            chrono_strat.computed_tokens += miss
            prev_tokens["Chronological RAG Window"] = tokens
            chrono_strat.step_records.append(
                {
                    "depth": turn_depth,
                    "tokens": tok_len,
                    "hit": hit,
                    "miss": miss,
                }
            )

            # -------------------------------------------------------------
            # Strategy C: Context-Engineer 5-Stage Pipeline
            # -------------------------------------------------------------
            ce_strat = metrics["Context-Engineer"]
            assembled = assemble_context(
                turns=active_turns,
                query=query_text,
                system_prompt=system_prompt,
                config=self.config,
            )

            tokens = self._tokens_for_messages(assembled.messages)
            tok_len = len(tokens)
            hit = _calculate_common_prefix(prev_tokens["Context-Engineer"], tokens)
            miss = tok_len - hit
            ce_strat.total_prompt_tokens += tok_len
            ce_strat.cache_hit_tokens += hit
            ce_strat.computed_tokens += miss
            prev_tokens["Context-Engineer"] = tokens
            ce_strat.step_records.append(
                {
                    "depth": turn_depth,
                    "tokens": tok_len,
                    "hit": hit,
                    "miss": miss,
                    "budget_plan": assembled.budget_plan,
                }
            )

        return metrics


def simulate_cache_performance(
    turns: list[Turn],
    system_prompt: str,
    config: ContextConfig | None = None,
) -> dict[str, StrategyMetrics]:
    """Runs a standard progression measuring multi-query follow-up prefix reuse.

    Simulates queries arriving at checkpoint depths and multiple follow-up queries
    at the final incident triage depth (depth 100).
    """
    ledger = PrefixCacheLedger(config=config)

    # Realistic incident investigation sequence:
    # Multiple queries at depth 100 where follow-ups investigate various aspects of root cause.
    queries = [
        (30, "What is the CPU usage across gateway proxies?"),
        (60, "Are failover procedures completing nominal sync?"),
        (100, "Provide full status summary of the cluster."),
        (100, "Which partition ended up carrying the blame?"),
        (100, "What error code was reported on controller registers?"),
        (100, "Were writes diverted away from the faulty partition?"),
    ]
    return ledger.simulate(turns=turns, queries=queries, system_prompt=system_prompt)
