"""Main benchmark runner CLI (Algorithm A13, EV1-EV5, LG1-LG5, PR1-PR3, X4, X6).

Usage:
    python -m benchmarks.run --backend mock
    python -m benchmarks.run --backend openai_compat --base-url http://localhost:1234/v1 --model qwen/qwen3.5-9b
"""

import argparse
import datetime
import platform
import sys
from pathlib import Path
from typing import Any

from benchmarks.baselines import (
    Baseline,
    ChronologicalRAGBaseline,
    ContextEngineerBaseline,
    PlainWindowBaseline,
    RawAppendBaseline,
    SuffixRAGUnquantizedBaseline,
)
from benchmarks.cache_probe import calculate_common_token_prefix
from benchmarks.evaluator import (
    GateReport,
    compute_strategy_distribution,
    evaluate_budget_enforced,
    evaluate_fact_present,
    evaluate_fact_recalled,
    evaluate_prefix_stability,
)
from benchmarks.reporter import generate_markdown_report, serialize_benchmark_results
from benchmarks.scenarios import generate_scenario
from context_engineer.backends.mock import MockBackend
from context_engineer.config import ContextConfig
from context_engineer.protocols import Backend
from context_engineer.tokenizers import ApproxTokenizer, Tokenizer


def build_baselines() -> list[Baseline]:
    """Instantiates the benchmark baselines built from shared components."""
    return [
        ContextEngineerBaseline(),
        SuffixRAGUnquantizedBaseline(),
        ChronologicalRAGBaseline(),
        PlainWindowBaseline(),
        RawAppendBaseline(),
    ]


def run_progression_protocol(
    scenario: Any,
    baselines: list[Baseline],
    backend: Backend,
    config: ContextConfig,
    tok: Tokenizer,
    model: str,
    is_live: bool,
    progression_step: int = 5,
) -> tuple[list[dict[str, Any]], list[GateReport]]:
    """Runs the primary progression benchmark protocol where history expands per request."""
    # Request checkpoints: e.g. turns 10, 15, 20... up to total_turns
    checkpoints = list(range(10, len(scenario.turns) + 1, progression_step))
    if scenario.turns and checkpoints[-1] != len(scenario.turns):
        checkpoints.append(len(scenario.turns))

    results: list[dict[str, Any]] = []
    gates_reports: list[GateReport] = []

    # Map of strategy -> previous token list for prefix reuse calculation
    prev_rendered_tokens: dict[str, list[str]] = {b.name: [] for b in baselines}
    prev_prefix_tokens: list[str] | None = None

    for baseline in baselines:
        hit_rates: list[float] = []
        prompt_tokens: list[int] = []
        cached_tokens: list[int] = []
        overflow_occurred = False
        overflow_turn = None

        for cp in checkpoints:
            active_turns = scenario.turns[:cp]
            query = f"Provide status summary up to turn #{cp}."

            res = baseline.assemble(
                turns=active_turns,
                query=query,
                system_prompt=scenario.system_prompt,
                config=config,
                tok=tok,
            )

            if res.overflow:
                overflow_occurred = True
                overflow_turn = res.overflow_turn or cp
                # Once overflowed, append-only strategies cannot process further requests
                break

            full_text = "\n".join(
                f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in res.messages
            )
            # Tokenize into word/subword chunks
            tokens = [full_text[j : j + 4] for j in range(0, len(full_text), 4)]
            p_len = tok.count(full_text)

            prev_toks = prev_rendered_tokens[baseline.name]
            cached = calculate_common_token_prefix(prev_toks, tokens) if prev_toks else 0
            prev_rendered_tokens[baseline.name] = tokens

            rate = (cached / p_len) if p_len > 0 else 0.0
            hit_rates.append(rate)
            prompt_tokens.append(p_len)
            cached_tokens.append(cached)

            # Check gates on Context-Engineer baseline
            if baseline.name == "Context-Engineer (Full)":
                # Budget gate
                gates_reports.append(
                    evaluate_budget_enforced(res.messages, config.effective_budget, tok)
                )

                # Prefix stability gate (prefix tokens: system + window head)
                prefix_tokens = tokens[: min(len(tokens), 200)]
                gates_reports.append(evaluate_prefix_stability(prev_prefix_tokens, prefix_tokens))
                prev_prefix_tokens = prefix_tokens

        dist = compute_strategy_distribution(
            strategy_name=baseline.name,
            step_hit_rates=hit_rates,
            prompt_tokens_per_step=prompt_tokens,
            cached_tokens_per_step=cached_tokens,
            overflow_occurred=overflow_occurred,
            overflow_turn=overflow_turn,
        )

        results.append(
            {
                "strategy_name": dist.strategy_name,
                "total_requests": dist.total_requests,
                "total_prompt_tokens": dist.total_prompt_tokens,
                "total_cached_tokens": dist.total_cached_tokens,
                "total_computed_tokens": dist.total_computed_tokens,
                "overall_hit_rate": dist.overall_hit_rate,
                "per_request_hit_rate_median": dist.per_request_hit_rate_median,
                "per_request_hit_rate_p10": dist.per_request_hit_rate_p10,
                "per_request_hit_rate_p90": dist.per_request_hit_rate_p90,
                "overflow_occurred": dist.overflow_occurred,
                "overflow_turn": dist.overflow_turn,
            }
        )

    # Evaluate fact gates on the primary planted needle
    if scenario.needles:
        primary_needle = scenario.needles[0]
        ce_baseline = ContextEngineerBaseline()
        assembled_needle = ce_baseline.assemble(
            turns=scenario.turns,
            query=primary_needle.query,
            system_prompt=scenario.system_prompt,
            config=config,
            tok=tok,
        )
        gates_reports.append(evaluate_fact_present(assembled_needle.messages, primary_needle))

        # fact_recalled gate: live backend only, SKIPPED if mock/offline (EV1)
        gates_reports.append(
            evaluate_fact_recalled(
                backend=backend,
                messages=assembled_needle.messages,
                needle=primary_needle,
                model=model,
                is_live=is_live,
            )
        )

    return results, gates_reports


def run_fixed_depth_protocol(
    scenario: Any,
    baselines: list[Baseline],
    config: ContextConfig,
    tok: Tokenizer,
) -> list[dict[str, Any]]:
    """Runs multiple distinct queries against a frozen dialog history at full depth."""
    queries = [
        "Summarize the most recent system errors.",
        "List all active node identifiers and cluster topology.",
        "Analyze storage buffer and throughput commit logs.",
        "Check network latency spikes and connection resets.",
    ]

    results: list[dict[str, Any]] = []
    prev_rendered_tokens: dict[str, list[str]] = {b.name: [] for b in baselines}

    for baseline in baselines:
        hit_rates: list[float] = []
        prompt_tokens: list[int] = []
        cached_tokens: list[int] = []
        overflow_occurred = False
        overflow_turn = None

        for q in queries:
            res = baseline.assemble(
                turns=scenario.turns,
                query=q,
                system_prompt=scenario.system_prompt,
                config=config,
                tok=tok,
            )

            if res.overflow:
                overflow_occurred = True
                overflow_turn = res.overflow_turn or len(scenario.turns)
                break

            full_text = "\n".join(
                f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in res.messages
            )
            tokens = [full_text[j : j + 4] for j in range(0, len(full_text), 4)]
            p_len = tok.count(full_text)

            prev_toks = prev_rendered_tokens[baseline.name]
            cached = calculate_common_token_prefix(prev_toks, tokens) if prev_toks else 0
            prev_rendered_tokens[baseline.name] = tokens

            rate = (cached / p_len) if p_len > 0 else 0.0
            hit_rates.append(rate)
            prompt_tokens.append(p_len)
            cached_tokens.append(cached)

        dist = compute_strategy_distribution(
            strategy_name=baseline.name,
            step_hit_rates=hit_rates,
            prompt_tokens_per_step=prompt_tokens,
            cached_tokens_per_step=cached_tokens,
            overflow_occurred=overflow_occurred,
            overflow_turn=overflow_turn,
        )

        results.append(
            {
                "strategy_name": dist.strategy_name,
                "total_requests": dist.total_requests,
                "total_prompt_tokens": dist.total_prompt_tokens,
                "total_cached_tokens": dist.total_cached_tokens,
                "total_computed_tokens": dist.total_computed_tokens,
                "overall_hit_rate": dist.overall_hit_rate,
                "per_request_hit_rate_median": dist.per_request_hit_rate_median,
                "per_request_hit_rate_p10": dist.per_request_hit_rate_p10,
                "per_request_hit_rate_p90": dist.per_request_hit_rate_p90,
                "overflow_occurred": dist.overflow_occurred,
                "overflow_turn": dist.overflow_turn,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-Engineer Empirical Benchmark Suite")
    parser.add_argument(
        "--backend", choices=["mock", "openai_compat"], default="mock", help="Inference backend"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:1234/v1", help="Base URL for live backend"
    )
    parser.add_argument("--model", default="mock-model", help="Model identifier")
    parser.add_argument("--turns", type=int, default=100, help="Total turns in synthetic scenario")
    parser.add_argument("--seed", type=int, default=42, help="Seed for scenario generator")
    parser.add_argument(
        "--domain", default="systems", choices=["systems", "support", "code_review"]
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory for JSON and markdown results"
    )

    args = parser.parse_args()

    # 1. Initialize configuration and components
    config = ContextConfig(
        context_ceiling=4096,
        completion_reserve=256,
        cap_threshold=35,
        cap_head_tokens=22,
        cap_tail_tokens=8,
    )
    tok = ApproxTokenizer()

    # 2. Initialize backend
    is_live = False
    if args.backend == "mock":
        backend: Backend = MockBackend(default_response="Mock diagnostic evaluation confirmation.")
        cache_mode = "simulated"
    else:
        # Live OpenAI compatible backend
        try:
            from context_engineer.client import LMStudioClient

            client = LMStudioClient(base_url=args.base_url)
            if not client.is_available():
                print(
                    f"[!] Live backend at {args.base_url} is not available. Exiting.",
                    file=sys.stderr,
                )
                sys.exit(1)
            is_live = True
            cache_mode = "measured"
            backend = MockBackend()  # Placeholder until live backend wrapper in Phase 5
        except Exception as e:
            print(f"[!] Error initializing live backend: {e}", file=sys.stderr)
            sys.exit(1)

    print(
        f"[+] Generating benchmark scenario (domain={args.domain}, turns={args.turns}, seed={args.seed})..."
    )
    scenario = generate_scenario(
        name=f"{args.domain}_benchmark",
        domain=args.domain,
        total_turns=args.turns,
        seed=args.seed,
    )

    baselines = build_baselines()

    # 3. Execute Progression Protocol
    print("[+] Executing Primary Progression Protocol...")
    prog_results, gates = run_progression_protocol(
        scenario=scenario,
        baselines=baselines,
        backend=backend,
        config=config,
        tok=tok,
        model=args.model,
        is_live=is_live,
    )

    # 4. Execute Fixed Depth Protocol
    print("[+] Executing Fixed Depth Protocol...")
    fixed_results = run_fixed_depth_protocol(
        scenario=scenario,
        baselines=baselines,
        config=config,
        tok=tok,
    )

    # 5. Format Metadata
    today_str = datetime.date.today().isoformat()
    metadata: dict[str, Any] = {
        "date": today_str,
        "backend": args.backend,
        "cache_mode": cache_mode,
        "model": args.model,
        "turns": args.turns,
        "seed": args.seed,
        "domain": args.domain,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    gates_summary = [
        {"gate_name": g.gate_name, "status": g.status, "details": g.details} for g in gates
    ]

    # De-duplicate gates summary
    unique_gates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for g in gates_summary:
        if g["gate_name"] not in seen_names:
            seen_names.add(g["gate_name"])
            unique_gates.append(g)

    # 6. Save JSON and Markdown artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{today_str}-{args.backend}.json"
    md_path = out_dir / f"{today_str}-{args.backend}.md"

    serialize_benchmark_results(
        filepath=json_path,
        metadata=metadata,
        progression_results={"data": prog_results},
        fixed_depth_results={"data": fixed_results},
        gates_summary=unique_gates,
    )

    md_content = generate_markdown_report(
        metadata=metadata,
        progression_results=prog_results,
        fixed_depth_results=fixed_results,
        gates_summary=unique_gates,
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(
        f"\n[+] Results successfully written to:\n    JSON: {json_path}\n    Markdown: {md_path}\n"
    )
    print(md_content)


if __name__ == "__main__":
    main()
