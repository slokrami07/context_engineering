"""Configuration limits and defaults for context-engineer."""

from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Central configuration for context limits, budget ceilings, and layer thresholds.

    Attributes:
        context_ceiling: Maximum token ceiling of the model context (e.g. 4096 or 8192).
        completion_reserve: Output token reservation reserved strictly for generation (e.g. 256).
        cap_threshold: Minimum tokens in a turn / tool output before capping is triggered (e.g. 35).
        cap_head_tokens: Tokens preserved at the beginning during head/tail truncation (e.g. 22).
        cap_tail_tokens: Tokens preserved at the end during head/tail truncation (e.g. 8).
        min_window_turns: Minimum contiguous recency turns to keep in the window if feasible.
        max_summary_tokens: Token budget allocated to the entity-free narrative summary.
        bm25_top_k: Number of dropped turns to retrieve via BM25 (default 1).
        bm25_min_score: Minimum BM25 score required for retrieval injection.
        model_name: Target model identifier (default: Qwen/Qwen2.5-7B-Instruct).
        lm_studio_base_url: Base URL for local LM Studio instance.
        system_prompt_reserve: Default reservation for system prompt when estimating.
    """

    context_ceiling: int = 4096
    completion_reserve: int = 256
    cap_threshold: int = 35
    cap_head_tokens: int = 22
    cap_tail_tokens: int = 8
    min_window_turns: int = 2
    max_summary_tokens: int = 150
    bm25_top_k: int = 1
    bm25_min_score: float = 0.1
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    system_prompt_reserve: int = 120

    @property
    def effective_budget(self) -> int:
        """Available budget for prompt tokens after reserving completion space."""
        return self.context_ceiling - self.completion_reserve
