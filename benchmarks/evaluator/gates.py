"""Evaluation gates for benchmark runs (Algorithm A13, EV1-EV5)."""

import dataclasses
import re
from typing import Literal

from benchmarks.scenarios.types import Needle
from context_engineer.backends import run_chat_sync
from context_engineer.protocols import Backend
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Message

GateStatus = Literal["PASSED", "FAILED", "SKIPPED"]


@dataclasses.dataclass(frozen=True, slots=True)
class GateReport:
    """Report for an evaluation gate."""

    gate_name: str
    status: GateStatus
    details: str


def evaluate_fact_present(messages: list[Message], needle: Needle) -> GateReport:
    """Gate 1: Verifies whether the planted fact/needle is present in the assembled context."""
    full_text = " ".join(m.content for m in messages)
    found = needle.key.lower() in full_text.lower()
    return GateReport(
        gate_name="fact_present",
        status="PASSED" if found else "FAILED",
        details=f"Needle '{needle.key}' present in assembled messages: {found}",
    )


def evaluate_budget_enforced(
    messages: list[Message],
    effective_ceiling: int,
    tok: Tokenizer,
) -> GateReport:
    """Gate 2 (I5): Verifies that total prompt tokens do not exceed the effective ceiling."""
    total_tokens = sum(tok.count(m.content) for m in messages)
    passed = total_tokens <= effective_ceiling
    return GateReport(
        gate_name="budget_enforced",
        status="PASSED" if passed else "FAILED",
        details=f"Prompt tokens: {total_tokens} <= ceiling: {effective_ceiling}",
    )


def evaluate_prefix_stability(
    prev_prefix_tokens: list[str] | None,
    curr_prefix_tokens: list[str],
) -> GateReport:
    """Gate 3 (I1/I2): Asserts prefix zone token stream stability across consecutive requests."""
    if prev_prefix_tokens is None:
        return GateReport(
            gate_name="prefix_stability",
            status="PASSED",
            details="Initial request (baseline established)",
        )

    # Prefix stability check: curr_prefix_tokens should start with or match prev_prefix_tokens
    shared = 0
    for a, b in zip(prev_prefix_tokens, curr_prefix_tokens, strict=False):
        if a == b:
            shared += 1
        else:
            break

    is_stable = shared >= min(len(prev_prefix_tokens), len(curr_prefix_tokens))
    return GateReport(
        gate_name="prefix_stability",
        status="PASSED" if is_stable else "FAILED",
        details=f"Shared prefix length: {shared} / min({len(prev_prefix_tokens)}, {len(curr_prefix_tokens)})",
    )


def evaluate_fact_recalled(
    backend: Backend,
    messages: list[Message],
    needle: Needle,
    model: str = "mock-model",
    is_live: bool = False,
    control_messages: list[Message] | None = None,
) -> GateReport:
    """Gate 4 (EV1-EV3): Evaluates empirical fact recall using a live backend.

    CRITICAL RULE (EV1): If the backend is offline or mock, this gate returns SKIPPED.
    It NEVER synthesizes or fabricates a model response.

    When live:
    - Sends prompt messages.
    - Evaluates normalized match of expected answer.
    - Rejects negations (e.g. 'It was NOT node-alpha-19').
    - Runs a negative control without retrieval which MUST fail.
    """
    if not is_live:
        return GateReport(
            gate_name="fact_recalled",
            status="SKIPPED",
            details="Offline or mock backend: live generation skipped to avoid fabricated results (EV1)",
        )

    # 1. Negative control run: without retrieval, model should not know the answer
    if control_messages:
        control_res = run_chat_sync(backend, control_messages, model=model)
        control_text = control_res.content.lower()
        if needle.expected_answer.lower() in control_text:
            return GateReport(
                gate_name="fact_recalled",
                status="FAILED",
                details="Negative control failed: model hallucinated or leaked the answer without retrieval (EV3)",
            )

    # 2. Main response evaluation
    res = run_chat_sync(backend, messages, model=model)
    resp_text = res.content.lower()

    # Check for presence of expected answer
    has_answer = needle.expected_answer.lower() in resp_text

    # Check for negation rejection: e.g. "not <key>", "never <key>"
    negation_pattern = (
        rf"\b(?:not|never|neither|except)\s+{re.escape(needle.expected_answer.lower())}"
    )
    is_negated = bool(re.search(negation_pattern, resp_text))

    if has_answer and not is_negated:
        return GateReport(
            gate_name="fact_recalled",
            status="PASSED",
            details=f"Answer '{needle.expected_answer}' correctly recalled without negation",
        )
    return GateReport(
        gate_name="fact_recalled",
        status="FAILED",
        details=f"Model response did not confirm expected answer '{needle.expected_answer}' (negated={is_negated})",
    )
