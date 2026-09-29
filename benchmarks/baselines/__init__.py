"""Baselines package for benchmark comparisons."""

from benchmarks.baselines.base import Baseline, BaselineResult
from benchmarks.baselines.chronological_rag import ChronologicalRAGBaseline
from benchmarks.baselines.context_engineer import ContextEngineerBaseline
from benchmarks.baselines.plain_window import PlainWindowBaseline
from benchmarks.baselines.raw_append import RawAppendBaseline
from benchmarks.baselines.suffix_rag import SuffixRAGUnquantizedBaseline

__all__ = [
    "Baseline",
    "BaselineResult",
    "ChronologicalRAGBaseline",
    "ContextEngineerBaseline",
    "PlainWindowBaseline",
    "RawAppendBaseline",
    "SuffixRAGUnquantizedBaseline",
]
