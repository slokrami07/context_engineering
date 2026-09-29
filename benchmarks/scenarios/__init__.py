"""Scenarios package for benchmark evaluation."""

from benchmarks.scenarios.generator import generate_scenario
from benchmarks.scenarios.types import BenchmarkScenario, Needle

__all__ = ["BenchmarkScenario", "Needle", "generate_scenario"]
