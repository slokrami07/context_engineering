"""Benchmark results reporter: serializes JSON outputs and generates markdown reports (Algorithm A13, X4, X6)."""

import json
from pathlib import Path
from typing import Any


def serialize_benchmark_results(
    filepath: Path,
    metadata: dict[str, Any],
    progression_results: dict[str, Any],
    fixed_depth_results: dict[str, Any],
    gates_summary: list[dict[str, Any]],
) -> None:
    """Serializes benchmark results to a formatted JSON file."""
    data = {
        "metadata": metadata,
        "progression_results": progression_results,
        "fixed_depth_results": fixed_depth_results,
        "gates_summary": gates_summary,
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_markdown_report(
    metadata: dict[str, Any],
    progression_results: list[dict[str, Any]],
    fixed_depth_results: list[dict[str, Any]],
    gates_summary: list[dict[str, Any]],
) -> str:
    """Generates an honest, GitHub-flavored markdown report from benchmark results."""
    lines: list[str] = []

    lines.append("# Empirical Benchmark Report: Context Management & Prefix Caching")
    lines.append("")
    lines.append(f"**Date:** {metadata.get('date', 'Unknown')}")
    lines.append(
        f"**Backend:** `{metadata.get('backend', 'unknown')}` ({metadata.get('cache_mode', 'simulated')})"
    )
    lines.append(f"**Model:** `{metadata.get('model', 'unknown')}`")
    lines.append(
        f"**Environment:** Python {metadata.get('python_version', '')} ({metadata.get('platform', '')})"
    )
    lines.append("")

    # 1. Progression Protocol Table
    lines.append("## 1. Primary Progression Protocol")
    lines.append(
        "Evaluates conversational progression where dialogue grows by 1-2 turns per user request."
    )
    lines.append(
        "Unlike static depth queries, this directly measures prefix stability under history expansion."
    )
    lines.append("")
    lines.append(
        "| Strategy | Total Prompt Tokens | Recomputed (KV Miss) | Cache Hit Ratio | Per-Req Hit Rate (Median) | p10 | p90 | Status |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for item in progression_results:
        strat = item["strategy_name"]
        total = item["total_prompt_tokens"]
        recomputed = item["total_computed_tokens"]
        overall = f"{item['overall_hit_rate'] * 100:.1f}%"
        med = f"{item['per_request_hit_rate_median'] * 100:.1f}%"
        p10 = f"{item['per_request_hit_rate_p10'] * 100:.1f}%"
        p90 = f"{item['per_request_hit_rate_p90'] * 100:.1f}%"
        status = (
            f"OVERFLOW (turn #{item['overflow_turn']})"
            if item.get("overflow_occurred")
            else "PASSED"
        )
        lines.append(
            f"| **{strat}** | {total:,} | {recomputed:,} | **{overall}** | {med} | {p10} | {p90} | {status} |"
        )
    lines.append("")

    # 2. Fixed Depth Queries Table
    lines.append("## 2. Fixed-Depth Queries (Multiple Queries at Frozen History)")
    lines.append(
        "Measures prefix reuse across consecutive queries issued at the same conversation depth."
    )
    lines.append("")
    lines.append(
        "| Strategy | Total Prompt Tokens | Recomputed Tokens | Cache Hit Ratio | Status |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for item in fixed_depth_results:
        strat = item["strategy_name"]
        total = item["total_prompt_tokens"]
        recomp = item["total_computed_tokens"]
        overall = f"{item['overall_hit_rate'] * 100:.1f}%"
        status = (
            f"OVERFLOW (turn #{item['overflow_turn']})"
            if item.get("overflow_occurred")
            else "PASSED"
        )
        lines.append(f"| **{strat}** | {total:,} | {recomp:,} | **{overall}** | {status} |")
    lines.append("")

    # 3. Build & Integrity Gates
    lines.append("## 3. Evaluation Gates")
    lines.append("")
    lines.append("| Gate Name | Status | Details |")
    lines.append("| :--- | :---: | :--- |")
    for g in gates_summary:
        icon = (
            "[PASS]"
            if g["status"] == "PASSED"
            else ("[SKIP]" if g["status"] == "SKIPPED" else "[FAIL]")
        )
        lines.append(f"| `{g['gate_name']}` | **{icon} {g['status']}** | {g['details']} |")
    lines.append("")

    # 4. Where Context-Engineer Loses (Honest Tradeoff Analysis X6)
    lines.append("## 4. Tradeoff Analysis: Where Context-Engineer Loses")
    lines.append(
        "A trustworthy library must document its engineering tradeoffs rather than presenting one-sided metrics:"
    )
    lines.append(
        "1. **Higher Total Prompt Tokens:** Context-Engineer preserves fixed reserves for system prompt, "
        "pinned invariant facts, and narrative summaries. This overhead means Context-Engineer consumes "
        "more *total* tokens than plain left-truncation while drastically cutting *recomputed* (cache miss) tokens."
    )
    lines.append(
        "2. **Reserve Slack:** Unused space in the fixed suffix and completion reserves is intentionally left as slack "
        "to prevent query-dependent prefix invalidation. Under tight context ceilings, this reduces the maximum turns "
        "retained in the active window."
    )
    lines.append(
        "3. **Lexical Paraphrase Misses with BM25:** If using BM25-only retrieval, questions with zero rare-token "
        "overlap with the original turn will fail recall. Use `HybridRetriever` with dense embeddings when semantic paraphrase "
        "recall is required."
    )
    lines.append(
        "4. **Mid-Session Pinning Cost:** Adding a `/pin` mid-dialogue intentionally invalidates the prefix from that turn forward, "
        "as required to guarantee invariant fact retention."
    )
    lines.append("")

    return "\n".join(lines)
