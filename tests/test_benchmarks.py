"""Tests verifying Phase 4 Benchmark Suite, Cache Probe, and Evaluation Gates."""

import json
import tempfile
from pathlib import Path

from benchmarks.baselines import (
    ChronologicalRAGBaseline,
    ContextEngineerBaseline,
    PlainWindowBaseline,
    RawAppendBaseline,
    SuffixRAGUnquantizedBaseline,
)
from benchmarks.cache_probe import calculate_common_token_prefix, run_cache_probe
from benchmarks.evaluator import (
    evaluate_fact_recalled,
)
from benchmarks.reporter import generate_markdown_report, serialize_benchmark_results
from benchmarks.run import (
    build_baselines,
    run_fixed_depth_protocol,
    run_progression_protocol,
)
from benchmarks.scenarios import generate_scenario
from context_engineer.backends.mock import MockBackend
from context_engineer.config import ContextConfig
from context_engineer.tokenizers import ApproxTokenizer
from context_engineer.types import Message


def test_scenario_generator_properties() -> None:
    """Verifies that scenario generator produces alternating roles and deterministic needles."""
    scenario1 = generate_scenario(name="test_1", domain="systems", total_turns=50, seed=123)
    scenario2 = generate_scenario(name="test_2", domain="systems", total_turns=50, seed=123)

    assert len(scenario1.turns) == 50
    # Determinism across same seed (PR3)
    assert [t.content for t in scenario1.turns] == [t.content for t in scenario2.turns]

    # Strictly alternating roles (PR2)
    for i, t in enumerate(scenario1.turns):
        expected_role = "user" if (i + 1) % 2 != 0 else "assistant"
        assert t.role == expected_role, f"Turn {t.id} expected role {expected_role}, got {t.role}"

    # Needles planted
    assert len(scenario1.needles) > 0
    for needle in scenario1.needles:
        # Needle should be in the turns
        needle_turn = next(t for t in scenario1.turns if t.id == needle.turn_id)
        assert needle.key in needle_turn.content


def test_cache_probe_calculation() -> None:
    """Verifies prefix calculation in cache probe."""
    tok1 = ["<system>", "hello", "world", "how", "are", "you"]
    tok2 = ["<system>", "hello", "world", "what", "is", "up"]
    shared = calculate_common_token_prefix(tok1, tok2)
    assert shared == 3  # "<system>", "hello", "world"

    backend = MockBackend()
    tok = ApproxTokenizer()
    reqs = [
        [Message("system", "Fixed instructions"), Message("user", "Hello")],
        [
            Message("system", "Fixed instructions"),
            Message("user", "Hello"),
            Message("assistant", "Hi"),
        ],
    ]
    summary = run_cache_probe(reqs, backend=backend, tok=tok)
    assert summary.total_prompt_tokens > 0
    assert summary.step_records[1].cached_tokens > 0
    assert not summary.is_measured  # Mock backend is simulated (LG4)


def test_offline_evaluator_skips_fact_recall_never_fabricates() -> None:
    """EV1: Verifies offline/mock backend marks fact_recalled as SKIPPED and never fabricates."""
    scenario = generate_scenario(total_turns=20, seed=42)
    needle = scenario.needles[0]
    backend = MockBackend()

    msgs = [Message("system", scenario.system_prompt), Message("user", needle.query)]
    report = evaluate_fact_recalled(
        backend=backend,
        messages=msgs,
        needle=needle,
        is_live=False,  # Offline mode
    )

    assert report.status == "SKIPPED"
    assert "EV1" in report.details
    assert "skipped to avoid fabricated results" in report.details


def test_all_baselines_run_with_shared_components() -> None:
    """LG1: Verifies that all baselines run cleanly with shared tokenizer and budget without hardcoding."""
    scenario = generate_scenario(total_turns=30, seed=42)
    config = ContextConfig(context_ceiling=2048, completion_reserve=256)
    tok = ApproxTokenizer()
    query = "Inspect system status."

    baselines = [
        PlainWindowBaseline(),
        ChronologicalRAGBaseline(),
        SuffixRAGUnquantizedBaseline(),
        ContextEngineerBaseline(),
        RawAppendBaseline(),
    ]

    for b in baselines:
        res = b.assemble(scenario.turns, query, scenario.system_prompt, config, tok)
        assert res.name == b.name
        assert not res.overflow
        assert res.prompt_tokens > 0
        assert len(res.messages) >= 2


def test_progression_protocol_and_reporter_end_to_end() -> None:
    """Verifies end-to-end progression protocol execution and artifact generation."""
    scenario = generate_scenario(total_turns=30, seed=42)
    config = ContextConfig(context_ceiling=2048, completion_reserve=256)
    tok = ApproxTokenizer()
    backend = MockBackend()
    baselines = build_baselines()

    prog_results, gates = run_progression_protocol(
        scenario=scenario,
        baselines=baselines,
        backend=backend,
        config=config,
        tok=tok,
        model="mock-model",
        is_live=False,
        progression_step=10,
    )

    assert len(prog_results) == len(baselines)
    ce_res = next(r for r in prog_results if "Context-Engineer" in r["strategy_name"])
    assert ce_res["overall_hit_rate"] > 0.0

    fixed_results = run_fixed_depth_protocol(
        scenario=scenario,
        baselines=baselines,
        config=config,
        tok=tok,
    )
    assert len(fixed_results) == len(baselines)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "results.json"
        metadata = {"backend": "mock", "date": "2026-09-30"}
        gates_summary = [
            {"gate_name": g.gate_name, "status": g.status, "details": g.details} for g in gates
        ]

        serialize_benchmark_results(
            filepath=json_path,
            metadata=metadata,
            progression_results={"data": prog_results},
            fixed_depth_results={"data": fixed_results},
            gates_summary=gates_summary,
        )

        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            assert "progression_results" in data

        md_text = generate_markdown_report(
            metadata=metadata,
            progression_results=prog_results,
            fixed_depth_results=fixed_results,
            gates_summary=gates_summary,
        )
        assert "Primary Progression Protocol" in md_text
        assert "Where Context-Engineer Loses" in md_text
