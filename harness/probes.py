"""Synthetic 100-turn incident management log generator with planted needles.

Simulates a high-scale production outage investigation with telemetry logs,
tool diagnostics, and a buried root-cause fact needle.
"""

from dataclasses import dataclass
from typing import Optional
import random

from context_engineer.types import Turn


@dataclass
class PlantedProbe:
    """Encapsulates a probe benchmark dataset with planted needles."""

    turns: list[Turn]
    planted_needle: str
    needle_turn_id: int
    query: str
    system_prompt: str


def generate_incident_turns(
    total_turns: int = 100,
    needle_turn_id: int = 14,
    needle_value: str = "shard: shard-19",
    seed: int = 42,
) -> PlantedProbe:
    """Generates a 100-turn synthetic DevOps incident management conversation.

    Args:
        total_turns: Total number of conversational turns to generate (default 100).
        needle_turn_id: 0-indexed turn where the critical needle fact is buried (default 14).
        needle_value: The specific root-cause identifier to plant.
        seed: Random seed for deterministic generation.

    Returns:
        PlantedProbe containing turns, planted needle, and query.
    """
    random.seed(seed)

    system_prompt = (
        "You are an expert site reliability engineering (SRE) assistant. "
        "Analyze the provided incident history and answer technical queries "
        "directly, concisely, and factually without hallucination."
    )

    query = "Which partition ended up carrying the blame?"

    turns: list[Turn] = []

    # Incident phases:
    # 0-5: Initial alert triggers & pager triage
    # 6-13: Network and gateway diagnostics
    # 14: Planted root-cause needle (Storage/partition diagnostics)
    # 15-30: Failover procedures and disk controller triage
    # 31-70: Rolling updates, traffic redirection, cache warming
    # 71-99: Post-incident review, SLO tracking, telemetry monitoring

    background_templates = [
        ("user", "What is the current p99 latency across the ingress proxies?"),
        ("assistant", "Ingress proxy telemetry indicates p99 latency elevated to 420ms across region us-east-1. Error rate 0.4%."),
        ("user", "Run health checks on upstream connection pools."),
        ("assistant", "Upstream pools connection status: 98.2% healthy. 4 pool sockets timed out during handshakes."),
        ("user", "Check memory consumption on cache cluster nodes."),
        ("assistant", "Cache cluster memory utilization steady at 64.1%. Eviction rates within nominal thresholds."),
        ("user", "Are there any pending lock contentions on the relational read replicas?"),
        ("assistant", "Relational read replicas report zero deadlock events. Replication lag is under 12ms."),
        ("user", "Check the latest deployment canary metrics."),
        ("assistant", "Canary deployment v2.14.0 is paused. Traffic split held at 5% pending error rate stabilization."),
        ("user", "Inspect cross-zone VPC transit peering throughput."),
        ("assistant", "Transit peering packet drop rate is 0.001%. Bandwidth ceiling operating at 42Gbps."),
    ]

    for i in range(total_turns):
        if i == needle_turn_id:
            # Plant the critical root cause needle!
            turn = Turn(
                id=i,
                role="assistant",
                content=(
                    "Deep partition telemetry inspection completed. "
                    "Storage layer write queue stalled completely in cluster zone B. "
                    f"Culprit root cause identified: {needle_value} suffered write queue deadlock "
                    "following an unrecoverable controller firmware timeout. "
                    "This partition ended up carrying the blame for the entire transaction rollback cascade."
                ),
                tool_output=(
                    f"DEBUG [storage_diag_worker]: Partition telemetry dump: {needle_value} error: 0xDEADBEEF. "
                    "Status: FAULT_QUARANTINED. Buffer saturation: 100%. "
                    "Controller registers: R0=0x0012, R1=0x889A, R2=0xFFFF. "
                    "Write barrier acknowledgment timeout exceeded 15000ms. "
                    f"All write operations diverted away from {needle_value}."
                ),
                pinned=False,
                metadata={"planted_needle": True, "needle_value": needle_value},
            )
        elif i == 0:
            turn = Turn(
                id=i,
                role="user",
                content="Alert fired: PagerDuty Sev-1 incident #9842: Distributed storage write availability degraded.",
                pinned=True,  # Pin the initial incident context
                metadata={"incident_id": 9842},
            )
        elif i == 1:
            turn = Turn(
                id=i,
                role="assistant",
                content="Acknowledged Sev-1. Initiating multi-cluster telemetry audit and diagnostic runbooks.",
                pinned=False,
            )
        else:
            tpl_role, tpl_content = background_templates[(i - 2) % len(background_templates)]
            # Add realistic varying noise / tool output to simulate realistic token loads
            tool_output = None
            if i % 3 == 0:
                tool_output = (
                    f"METRIC_SAMPLE[t={i}]: cpu_user={random.uniform(20.0, 75.0):.1f}% "
                    f"mem_avail={random.randint(12000, 48000)}MB "
                    f"iops={random.randint(2000, 18000)} "
                    f"tcp_retrans={random.randint(0, 12)} "
                    f"status=ACTIVE_OK"
                )

            turn = Turn(
                id=i,
                role=tpl_role,
                content=f"[Step {i}] {tpl_content}",
                tool_output=tool_output,
                pinned=False,
            )

        turns.append(turn)

    return PlantedProbe(
        turns=turns,
        planted_needle="shard-19",
        needle_turn_id=needle_turn_id,
        query=query,
        system_prompt=system_prompt,
    )
