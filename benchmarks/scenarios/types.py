"""Types and dataclasses for benchmark scenarios."""

import dataclasses

from context_engineer.types import Turn


@dataclasses.dataclass(frozen=True, slots=True)
class Needle:
    """A planted fact inside a benchmark history."""

    key: str
    value: str
    turn_id: int
    query: str
    paraphrased_query: str
    negative_query: str
    expected_answer: str


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """A synthetic benchmark scenario with alternating history and planted probes."""

    name: str
    domain: str
    system_prompt: str
    turns: list[Turn]
    needles: list[Needle]
    distractors: list[str] = dataclasses.field(default_factory=list)
