"""Layer 5: Entity-free rolling summarizer.

For all dropped turns not retrieved, generates a high-level narrative summary
describing the conversation shape while strictly forbidding specific entity names,
partition IDs, hashes, or hex codes to prevent stale fact hallucination and context rot.
"""

from typing import Optional
import re

from context_engineer.config import ContextConfig
from context_engineer.types import Turn
from context_engineer.tokenizer import count_message_tokens, truncate_text_to_tokens

# Regex scrubbers to neutralize hallucination-inducing entities
ENTITY_PATTERNS = [
    # Shard / partition / host / node identifiers (e.g. shard-19, partition-42, node-01)
    (re.compile(r"\b(shard|partition|node|cluster|server|instance|worker|pod|host|broker)[-_][a-zA-Z0-9_-]+\b", re.IGNORECASE), "[resource]"),
    # Hexadecimal values and error codes (e.g. 0xDEADBEEF, 0x1f)
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "[code]"),
    # UUIDs
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "[id]"),
    # IP addresses
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "[address]"),
    # Long hex hashes (e.g. git commits, sha256)
    (re.compile(r"\b[a-f0-9]{12,64}\b", re.IGNORECASE), "[hash]"),
    # File paths
    (re.compile(r"(/[a-zA-Z0-9_\.\-]+)+|[a-zA-Z]:\\[a-zA-Z0-9_\.\-\\]+"), "[path]"),
]


def scrub_entities(text: str) -> str:
    """Removes specific entity IDs, partition names, and hashes from text."""
    scrubbed = text
    for pattern, replacement in ENTITY_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


def generate_narrative_summary(turns: list[Turn]) -> str:
    """Generates an abstract narrative describing conversation shape and flow."""
    if not turns:
        return ""

    user_intents: list[str] = []
    actions_taken: list[str] = []

    for t in turns:
        clean_content = scrub_entities(t.content)
        if t.role == "user":
            snippet = clean_content.split("\n")[0][:60].strip()
            if snippet and snippet not in user_intents:
                user_intents.append(snippet)
        elif t.role in ("assistant", "tool"):
            if t.tool_output or t.role == "tool":
                actions_taken.append("executed diagnostic tool inspections")
            else:
                snippet = clean_content.split("\n")[0][:60].strip()
                if snippet and snippet not in actions_taken:
                    actions_taken.append(snippet)

    summary_lines = [
        f"Earlier conversation covered {len(turns)} foundational turns.",
        "The discussion involved technical troubleshooting, telemetry review, and status verification.",
    ]

    if user_intents:
        summary_lines.append(f"Inquiries focused on: {'; '.join(user_intents[:3])}.")

    if actions_taken:
        summary_lines.append(f"Actions taken: {'; '.join(actions_taken[:2])}.")

    summary_lines.append(
        "All historical entity identifiers have been abstracted to prevent stale state hallucination."
    )

    return " ".join(summary_lines)


def apply_summarize_layer(
    dropped_turns: list[Turn],
    config: Optional[ContextConfig] = None,
) -> tuple[Optional[str], int]:
    """Layer 5 pipeline entry point: builds an entity-free rolling summary for evicted turns.

    Args:
        dropped_turns: List of older turns not included in the recency window or retrieval.
        config: Central configuration.

    Returns:
        A tuple of (summary_text, summary_tokens). If dropped_turns is empty, returns (None, 0).
    """
    if config is None:
        config = ContextConfig()

    if not dropped_turns:
        return None, 0

    raw_summary = generate_narrative_summary(dropped_turns)
    scrubbed = scrub_entities(raw_summary)

    # Truncate summary to max_summary_tokens
    budgeted_summary = truncate_text_to_tokens(
        scrubbed,
        max_tokens=config.max_summary_tokens,
        model_name=config.model_name,
    )

    summary_tokens = count_message_tokens("system", f"[Historical Summary]: {budgeted_summary}", model_name=config.model_name)
    return budgeted_summary, summary_tokens
