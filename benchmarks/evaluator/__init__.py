"""Evaluator package for benchmark runs."""

from benchmarks.evaluator.gates import (
    GateReport,
    evaluate_budget_enforced,
    evaluate_fact_present,
    evaluate_fact_recalled,
    evaluate_prefix_stability,
)
from benchmarks.evaluator.metrics import (
    StrategyDistribution,
    compute_strategy_distribution,
    percentile,
)

__all__ = [
    "GateReport",
    "StrategyDistribution",
    "compute_strategy_distribution",
    "evaluate_budget_enforced",
    "evaluate_fact_present",
    "evaluate_fact_recalled",
    "evaluate_prefix_stability",
    "percentile",
]
