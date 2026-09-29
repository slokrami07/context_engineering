"""Central configuration, validation, and cap policies for context-engineer."""

from dataclasses import dataclass, field
from typing import Any, Literal

from context_engineer.errors import ConfigError


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Configuration for inference server backends (e.g. LM Studio, vLLM, Ollama)."""

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class TurnCapRule:
    """Individual capping threshold and strategy for specific roles or tool results."""

    threshold: int = 400
    head_tokens: int = 250
    tail_tokens: int = 100
    strategy: Literal["json", "lines", "token_slice"] = "token_slice"

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ConfigError(f"TurnCapRule threshold must be positive, got {self.threshold}")
        if self.head_tokens < 1:
            raise ConfigError(f"TurnCapRule head_tokens must be >= 1, got {self.head_tokens}")
        if self.tail_tokens < 1:
            raise ConfigError(f"TurnCapRule tail_tokens must be >= 1, got {self.tail_tokens}")
        if self.head_tokens + self.tail_tokens > self.threshold:
            raise ConfigError(
                f"head_tokens ({self.head_tokens}) + tail_tokens ({self.tail_tokens}) "
                f"cannot exceed threshold ({self.threshold})"
            )


@dataclass(frozen=True, slots=True)
class CapPolicy:
    """Configurable capping policy applied per role, tool name, or turn kind."""

    default_tool_rule: TurnCapRule = field(
        default_factory=lambda: TurnCapRule(threshold=400, head_tokens=250, tail_tokens=100)
    )
    cap_user_messages: bool = False
    user_rule: TurnCapRule | None = None
    tool_overrides: dict[str, TurnCapRule] = field(default_factory=dict)

    def for_turn(self, turn: Any) -> TurnCapRule | None:
        """Determines the active TurnCapRule for a given Turn."""
        if getattr(turn, "pinned", False):
            return None

        # Check tool results or tool turns
        kind = getattr(turn, "kind", "message")
        role = getattr(turn, "role", "user")
        name = getattr(turn, "name", None)

        if kind == "tool_result" or role == "tool" or getattr(turn, "tool_output", None):
            if name and name in self.tool_overrides:
                return self.tool_overrides[name]
            return self.default_tool_rule

        if role == "user" and self.cap_user_messages:
            return self.user_rule or TurnCapRule(threshold=1500, head_tokens=1000, tail_tokens=400)

        return None


def default_cap_policy() -> CapPolicy:
    """Returns standard default cap policy (caps tool outputs at 400 tokens, leaves messages intact)."""
    return CapPolicy()


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Central configuration for context limits, budget ceilings, and layer thresholds.

    Attributes:
        context_ceiling: Maximum token ceiling of the model context (e.g. 4096 or 8192).
        completion_reserve: Output token reservation reserved strictly for generation (e.g. 256).
        window_chunk_tokens: Quantization step size for the window boundary (None = 0.20 * window_budget).
        min_recent_turns: Minimum contiguous recency turns to keep in the window if feasible.
        require_user_start: If True, window boundary is adjusted to start on a user turn.
        merge_consecutive_roles: Merges adjacent same-role messages when rendering.
        suffix_reserve_tokens: Tokens reserved for the dynamic suffix (retrieval + query).
        max_summary_tokens: Token budget allocated to the entity-free narrative summary.
        retrieval_top_k: Number of dropped turns to retrieve via BM25 / hybrid search.
        budget_safety_margin: Safety margin percentage deducted from available budget.
        priming_tokens: Reservation for chat template prompt priming (<|im_start|>assistant).
        redaction_mode: Entity scrubbing mode ('none', 'light', 'strict').
        summary_placement: Summary insertion location ('system', 'user', 'fused').
        tokenizer: Tokenizer spec identifier ('approx', 'tiktoken:<enc>', 'hf:<repo>').
        strict_tokenizer: If True, raises TokenizerError instead of falling back to approx.
        truncate_query: If True, truncates oversized user query instead of raising ContextBudgetError.
        cap_policy: Central capping policy for tool and user turns.
    """

    context_ceiling: int = 4096
    completion_reserve: int = 256
    window_chunk_tokens: int | None = None
    min_recent_turns: int = 4
    require_user_start: bool = True
    merge_consecutive_roles: bool = True
    suffix_reserve_tokens: int = 768
    max_summary_tokens: int = 256
    retrieval_top_k: int = 3
    budget_safety_margin: float = 0.02
    priming_tokens: int = 3
    redaction_mode: Literal["none", "light", "strict"] = "light"
    summary_placement: Literal["system", "user", "fused"] = "fused"
    tokenizer: str = "approx"
    strict_tokenizer: bool = False
    truncate_query: bool = False
    cap_policy: CapPolicy = field(default_factory=default_cap_policy)

    # Legacy fields (maintained for backward compatibility with v0.1.0 callers)
    cap_threshold: int = 35
    cap_head_tokens: int = 22
    cap_tail_tokens: int = 8
    bm25_top_k: int = 1
    bm25_min_score: float = 0.1
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lm_studio_base_url: str = "http://localhost:1234/v1"
    system_prompt_reserve: int = 120

    def __post_init__(self) -> None:
        """Validates configuration invariants and prevents invalid token parameters."""
        if self.context_ceiling <= 0:
            raise ConfigError(f"context_ceiling must be positive, got {self.context_ceiling}")
        if self.completion_reserve <= 0:
            raise ConfigError(f"completion_reserve must be positive, got {self.completion_reserve}")
        if self.completion_reserve >= self.context_ceiling:
            raise ConfigError(
                f"completion_reserve ({self.completion_reserve}) cannot exceed or equal "
                f"context_ceiling ({self.context_ceiling})"
            )
        if self.suffix_reserve_tokens <= 0:
            raise ConfigError(
                f"suffix_reserve_tokens must be positive, got {self.suffix_reserve_tokens}"
            )
        if self.max_summary_tokens <= 0:
            raise ConfigError(f"max_summary_tokens must be positive, got {self.max_summary_tokens}")
        if self.retrieval_top_k < 1:
            raise ConfigError(f"retrieval_top_k must be at least 1, got {self.retrieval_top_k}")
        if self.min_recent_turns < 1:
            raise ConfigError(f"min_recent_turns must be at least 1, got {self.min_recent_turns}")
        if not (0.0 <= self.budget_safety_margin < 1.0):
            raise ConfigError(
                f"budget_safety_margin must be in range [0.0, 1.0), got {self.budget_safety_margin}"
            )
        if self.redaction_mode not in ("none", "light", "strict"):
            raise ConfigError(
                f"Invalid redaction_mode: {self.redaction_mode!r}. Must be 'none', 'light', or 'strict'."
            )
        if self.summary_placement not in ("system", "user", "fused"):
            raise ConfigError(
                f"Invalid summary_placement: {self.summary_placement!r}. Must be 'system', 'user', or 'fused'."
            )

        # Legacy cap limits validation and policy synchronization
        if self.cap_threshold <= 0:
            raise ConfigError(f"cap_threshold must be positive, got {self.cap_threshold}")
        if self.cap_head_tokens < 1 or self.cap_tail_tokens < 1:
            raise ConfigError("cap_head_tokens and cap_tail_tokens must be >= 1")
        if self.cap_head_tokens + self.cap_tail_tokens > self.cap_threshold:
            raise ConfigError(
                f"cap_head_tokens ({self.cap_head_tokens}) + cap_tail_tokens ({self.cap_tail_tokens}) "
                f"cannot exceed cap_threshold ({self.cap_threshold})"
            )

        if self.cap_threshold != 400 and self.cap_policy == default_cap_policy():
            object.__setattr__(
                self,
                "cap_policy",
                CapPolicy(
                    default_tool_rule=TurnCapRule(
                        threshold=self.cap_threshold,
                        head_tokens=self.cap_head_tokens,
                        tail_tokens=self.cap_tail_tokens,
                    )
                ),
            )

        # Budget headroom validation
        effective = self.context_ceiling - self.completion_reserve
        if (
            self.suffix_reserve_tokens == 768
            and self.max_summary_tokens == 256
            and effective < 1027
        ):
            # Auto-scale default reserves proportionally for custom small ceilings
            scaled_suffix = max(10, int(effective * 0.25))
            scaled_summary = max(10, int(effective * 0.15))
            object.__setattr__(self, "suffix_reserve_tokens", scaled_suffix)
            object.__setattr__(self, "max_summary_tokens", scaled_summary)

        fixed_minimum_reserves = (
            self.suffix_reserve_tokens + self.max_summary_tokens + self.priming_tokens
        )
        if fixed_minimum_reserves >= effective:
            raise ConfigError(
                f"Fixed reserves ({fixed_minimum_reserves}t) exceed or consume entire effective budget "
                f"({effective}t). Increase context_ceiling or decrease suffix/summary reserves."
            )

    @property
    def effective_budget(self) -> int:
        """Available budget for prompt tokens after reserving completion space."""
        return self.context_ceiling - self.completion_reserve
