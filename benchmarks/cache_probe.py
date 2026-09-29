"""Cache probe: measures KV cache reuse and prefix stability across consecutive turns (Algorithm A12)."""

import dataclasses

from context_engineer.backends import run_chat_sync
from context_engineer.protocols import Backend, CacheStats
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message


@dataclasses.dataclass(frozen=True, slots=True)
class ProbeStepResult:
    """Result of a single probe request."""

    step_index: int
    prompt_tokens: int
    cached_tokens: int
    computed_tokens: int
    cache_hit_rate: float
    is_measured: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CacheProbeSummary:
    """Aggregated results across multiple probe turns."""

    total_prompt_tokens: int
    total_cached_tokens: int
    total_computed_tokens: int
    overall_hit_rate: float
    is_measured: bool
    step_records: list[ProbeStepResult]


def calculate_common_token_prefix(tok1: list[str], tok2: list[str]) -> int:
    """Calculates the exact length of the shared token prefix between two token streams."""
    shared = 0
    for a, b in zip(tok1, tok2, strict=False):
        if a == b:
            shared += 1
        else:
            break
    return shared


def run_cache_probe(
    requests: list[list[Message]],
    backend: Backend,
    tok: Tokenizer,
    model: str = "mock-model",
) -> CacheProbeSummary:
    """Sends requests sequentially to the backend, measuring or simulating prefix cache reuse.

    Args:
        requests: List of message lists representing turns t, t+1, t+2...
        backend: Backend implementation (MockBackend or live backend).
        tok: Tokenizer for token-level prefix calculations.
        model: Model identifier.

    Returns:
        CacheProbeSummary containing per-step and aggregated cache statistics.
    """
    if not requests:
        return CacheProbeSummary(
            total_prompt_tokens=0,
            total_cached_tokens=0,
            total_computed_tokens=0,
            overall_hit_rate=0.0,
            is_measured=False,
            step_records=[],
        )

    step_records: list[ProbeStepResult] = []
    total_prompt = 0
    total_cached = 0
    total_computed = 0
    any_measured = False

    prev_tokens: list[str] = []

    for i, msgs in enumerate(requests):
        chat_res = run_chat_sync(backend, msgs, model=model)
        full_text = "\n".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in msgs)
        tokens = [full_text[j : j + 4] for j in range(0, len(full_text), 4)]  # Approximate tokens
        prompt_len = tok.count(full_text)

        stats: CacheStats | None = chat_res.cache

        if stats is not None and stats.measured:
            any_measured = True
            cached = stats.cached_tokens
            computed = max(0, prompt_len - cached)
            hit_rate = stats.cache_hit_rate
            is_meas = True
        else:
            # Simulated ideal longest-common-prefix (LG4)
            cached = calculate_common_token_prefix(prev_tokens, tokens) if prev_tokens else 0
            computed = prompt_len - cached
            hit_rate = (cached / prompt_len) if prompt_len > 0 else 0.0
            is_meas = False

        prev_tokens = tokens
        total_prompt += prompt_len
        total_cached += cached
        total_computed += computed

        step_records.append(
            ProbeStepResult(
                step_index=i,
                prompt_tokens=prompt_len,
                cached_tokens=cached,
                computed_tokens=computed,
                cache_hit_rate=hit_rate,
                is_measured=is_meas,
            )
        )

    overall_hit_rate = (total_cached / total_prompt) if total_prompt > 0 else 0.0

    return CacheProbeSummary(
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        total_computed_tokens=total_computed,
        overall_hit_rate=overall_hit_rate,
        is_measured=any_measured,
        step_records=step_records,
    )
