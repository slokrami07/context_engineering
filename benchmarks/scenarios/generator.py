"""Scenario generators for benchmark evaluation.

Implements procedural generation of long-turn dialogs across multiple domains:
- Operations / distributed systems
- Customer support
- Code review / software engineering

Features:
- Deterministic seeding via random.Random(seed) (PR3)
- Strictly alternating user/assistant roles (PR2)
- Needles planted at configurable depths (5%, 25%, 50%, 75%) (A13)
- Near-duplicate distractors (e.g. partition-19 vs partition-91 vs partition-07) (EV3)
- Paraphrased queries with zero rare-token overlap (EV2, A13)
- Multi-needle support (A13)
"""

import random

from benchmarks.scenarios.types import BenchmarkScenario, Needle
from context_engineer.types import Turn

# Background message templates across domains
DOMAIN_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "systems": {
        "user": [
            "Check cluster status on node {node_id}.",
            "Inspect memory utilization and disk buffer pools on host {node_id}.",
            "Run health probe against service instance {node_id}.",
            "Analyze write latency logs on replica group {node_id}.",
            "Validate heartbeat ping response on worker {node_id}.",
        ],
        "assistant": [
            "Node {node_id} reported 99.8% availability with steady load index {metric}.",
            "Buffer pools on host {node_id} operating within nominal range of {metric}MB.",
            "Service instance {node_id} responded to synthetic probe in {metric}ms.",
            "Replica group {node_id} committed 450 transactions/sec; average latency {metric}ms.",
            "Worker {node_id} heartbeat confirmed active; peer connections intact.",
        ],
    },
    "support": {
        "user": [
            "Customer #{node_id} submitted a ticket regarding billing cycle discrepancies.",
            "Review refund authorization request for transaction #{node_id}.",
            "Account #{node_id} requested two-factor authentication reset.",
            "Verify shipping tracking update for invoice #{node_id}.",
            "Inquire about enterprise license upgrade tier for client #{node_id}.",
        ],
        "assistant": [
            "Billing breakdown verified for customer #{node_id}; invoice balanced with ledger {metric}.",
            "Refund #{node_id} approved under standard return policy clause {metric}.",
            "Authentication verification completed for account #{node_id}; reset link dispatched.",
            "Tracking #{node_id} scanned at central distribution hub with status code {metric}.",
            "Enterprise tier proposal generated for client #{node_id} with discount code {metric}.",
        ],
    },
    "code_review": {
        "user": [
            "Please review pull request #{node_id} touching authentication middlewares.",
            "Examine static analysis output for commit hash #{node_id}.",
            "Check test coverage report on database connection pool module #{node_id}.",
            "Inspect memory benchmark profiles for serialization parser #{node_id}.",
            "Audit security advisory scan results on dependency package #{node_id}.",
        ],
        "assistant": [
            "PR #{node_id} clean: no regressions detected, lint passed with rating {metric}.",
            "Static analysis for #{node_id} reported zero high-severity findings.",
            "Branch #{node_id} has 94.2% unit test branch coverage across {metric} tests.",
            "Parser #{node_id} allocated 12KB per request with zero heap fragmentation.",
            "Dependency #{node_id} verified against CVE database; all hashes valid.",
        ],
    },
}


def generate_scenario(
    name: str = "systems_progression",
    domain: str = "systems",
    total_turns: int = 100,
    seed: int = 42,
    needle_depth_pcts: list[float] | None = None,
) -> BenchmarkScenario:
    """Generates a synthetic scenario with strictly alternating roles and planted needles.

    Args:
        name: Name of the scenario.
        domain: Domain template ("systems", "support", or "code_review").
        total_turns: Total number of turns to generate (must be >= 10).
        seed: Random seed for deterministic reproducibility (PR3).
        needle_depth_pcts: Percentile depths (0.0 to 1.0) at which to plant needles.
                           Defaults to [0.05, 0.25, 0.50, 0.75].

    Returns:
        BenchmarkScenario containing turns, needles, and near-duplicate distractors.
    """
    rng = random.Random(seed)
    depths = needle_depth_pcts if needle_depth_pcts is not None else [0.05, 0.25, 0.50, 0.75]

    templates = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["systems"])
    user_templates = templates["user"]
    asst_templates = templates["assistant"]

    turns: list[Turn] = []
    needles: list[Needle] = []

    # Calculate turn IDs for needles (ensuring they land on assistant turns, even turn IDs)
    target_turn_ids: list[int] = []
    for d in depths:
        idx = max(2, int(total_turns * d))
        # Ensure it is an assistant turn (even id in 1-indexed alternating dialog)
        if idx % 2 != 0:
            idx = min(total_turns, idx + 1)
        target_turn_ids.append(idx)

    # Near-duplicate identifier distractors to test precision (EV3)
    target_identifier = "node-alpha-19"
    distractors = ["node-alpha-91", "node-alpha-07", "node-alpha-29", "node-alpha-18"]

    # Pre-generate needles metadata
    needle_specs = [
        (
            target_identifier,
            "experienced write queue buffer deadlock causing replication timeout",
            "Which node suffered write queue buffer deadlock and failed replication?",
            "What compute unit had serialization pipeline lockups during cluster writes?",
            "Which node operated normally without write deadlock or cluster latency?",
        ),
        (
            "node-beta-42",
            "exceeded max connection quota triggering socket leak panic",
            "Which node exceeded max connection quota and panicked?",
            "Identify the machine that exhausted networking sockets and crashed.",
            "Which node experienced zero networking socket leaks?",
        ),
        (
            "node-gamma-88",
            "corrupted persistent commit log during unexpected power cycling",
            "Which node corrupted persistent commit log during power cycling?",
            "What storage server had write journal degradation during electrical reset?",
            "Which node completed persistent log validation with zero errors?",
        ),
    ]

    needle_map: dict[int, Needle] = {}
    for i, t_id in enumerate(target_turn_ids):
        spec = needle_specs[i % len(needle_specs)]
        ident, fact, query, para_query, neg_query = spec
        needle_map[t_id] = Needle(
            key=ident,
            value=fact,
            turn_id=t_id,
            query=query,
            paraphrased_query=para_query,
            negative_query=neg_query,
            expected_answer=ident,
        )
        needles.append(needle_map[t_id])

    # Build alternating turns
    current_distractor_idx = 0
    for turn_id in range(1, total_turns + 1):
        role = "user" if turn_id % 2 != 0 else "assistant"

        if turn_id in needle_map:
            # Planted needle turn
            needle = needle_map[turn_id]
            content = f"Critical incident diagnostics confirmed that {needle.key} {needle.value}."
        elif turn_id % 2 == 0 and current_distractor_idx < len(distractors) and turn_id % 7 == 0:
            # Plant near-duplicate distractor (EV3)
            distractor_ident = distractors[current_distractor_idx]
            current_distractor_idx += 1
            content = (
                f"Routine audit verified that {distractor_ident} completed nominal maintenance cycle "
                f"with buffer pool clearance and zero errors."
            )
        else:
            # Procedural background turn
            node_id = f"node-{rng.randint(100, 999)}"
            metric = rng.randint(10, 500)
            tmpl_list = user_templates if role == "user" else asst_templates
            template = rng.choice(tmpl_list)
            content = template.format(node_id=node_id, metric=metric)

        turns.append(Turn(id=turn_id, role=role, content=content))

    system_prompt = (
        f"You are a helpful and precise technical assistant analyzing {domain} records. "
        "Always cite specific identifiers when answering."
    )

    return BenchmarkScenario(
        name=name,
        domain=domain,
        system_prompt=system_prompt,
        turns=turns,
        needles=needles,
        distractors=distractors,
    )
