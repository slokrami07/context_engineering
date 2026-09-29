"""Metrics calculation and percentile distributions for benchmark evaluations (LG2, LG3, A13)."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class StrategyDistribution:
    """Distribution metrics for a strategy across progression steps."""

    strategy_name: str
    total_requests: int
    total_prompt_tokens: int
    total_cached_tokens: int
    total_computed_tokens: int
    overall_hit_rate: float
    per_request_hit_rate_median: float
    per_request_hit_rate_p10: float
    per_request_hit_rate_p90: float
    overflow_occurred: bool = False
    overflow_turn: int | None = None


def percentile(data: list[float], p: float) -> float:
    """Computes the p-th percentile of a sorted float list (p between 0.0 and 1.0)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def compute_strategy_distribution(
    strategy_name: str,
    step_hit_rates: list[float],
    prompt_tokens_per_step: list[int],
    cached_tokens_per_step: list[int],
    overflow_occurred: bool = False,
    overflow_turn: int | None = None,
) -> StrategyDistribution:
    """Computes median, p10, p90, and cumulative metrics across progression steps."""
    total_prompt = sum(prompt_tokens_per_step)
    total_cached = sum(cached_tokens_per_step)
    total_computed = total_prompt - total_cached
    overall_rate = (total_cached / total_prompt) if total_prompt > 0 else 0.0

    med = percentile(step_hit_rates, 0.50) if step_hit_rates else 0.0
    p10 = percentile(step_hit_rates, 0.10) if step_hit_rates else 0.0
    p90 = percentile(step_hit_rates, 0.90) if step_hit_rates else 0.0

    return StrategyDistribution(
        strategy_name=strategy_name,
        total_requests=len(step_hit_rates),
        total_prompt_tokens=total_prompt,
        total_cached_tokens=total_cached,
        total_computed_tokens=total_computed,
        overall_hit_rate=overall_rate,
        per_request_hit_rate_median=med,
        per_request_hit_rate_p10=p10,
        per_request_hit_rate_p90=p90,
        overflow_occurred=overflow_occurred,
        overflow_turn=overflow_turn,
    )
