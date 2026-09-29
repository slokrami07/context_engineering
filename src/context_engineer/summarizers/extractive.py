"""Extractive domain-agnostic summarizer implementing the Summarizer protocol."""

import re
from collections.abc import Sequence

from context_engineer.protocols import Redactor, Summarizer, Tokenizer
from context_engineer.redaction.redactor import RegexRedactor
from context_engineer.types import Turn


class ExtractiveSummarizer(Summarizer):
    """Deterministic, domain-agnostic extractive summarizer.

    Extracts high-salience statements across evicted turns without injecting
    fabricated domain boilerplate or turn counts into the generated summary.
    """

    id: str = "extractive"
    version: str = "1.0.0"

    def __init__(self, redactor: Redactor | None = None) -> None:
        self.redactor = redactor or RegexRedactor(mode="light")

    def _split_sentences(self, text: str) -> list[str]:
        """Splits text into clean sentences."""
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def summarize(
        self,
        dropped: Sequence[Turn],
        *,
        max_tokens: int,
        tok: Tokenizer,
    ) -> str:
        """Generates an abstract narrative summary for evicted turns."""
        if not dropped or max_tokens <= 0:
            return ""

        # Collect salient statements
        first_user_goal: str | None = None
        statements: list[str] = []

        for turn in dropped:
            clean_text = self.redactor.redact(turn.content)
            sentences = self._split_sentences(clean_text)
            if not sentences:
                continue

            if turn.role == "user" and first_user_goal is None:
                first_user_goal = sentences[0]

            # Collect informative sentences (questions, decisions, state changes)
            for s in sentences:
                if len(s) < 15:
                    continue
                # Informative heuristics: contains questions, colons, numbers, or action verbs
                has_signal = (
                    "?" in s
                    or ":" in s
                    or any(c.isdigit() for c in s)
                    or any(
                        w in s.lower()
                        for w in (
                            "error",
                            "failed",
                            "verified",
                            "started",
                            "completed",
                            "requested",
                        )
                    )
                )
                if has_signal and s not in statements:
                    statements.append(s)

        # Assemble summary components chronologically
        summary_sentences: list[str] = []
        if first_user_goal:
            summary_sentences.append(f"Initial objective: {first_user_goal}")

        # Greedily include most recent salient statements that fit within token budget
        budget_for_statements = max(
            20, max_tokens - (tok.count(summary_sentences[0]) if summary_sentences else 0)
        )

        # Select from recent statements backwards, then sort chronologically
        selected: list[str] = []
        used_tokens = 0
        for s in reversed(statements):
            cost = tok.count(s) + 1
            if used_tokens + cost <= budget_for_statements:
                selected.append(s)
                used_tokens += cost
            else:
                break

        summary_sentences.extend(reversed(selected))

        if not summary_sentences:
            return "Earlier context discussed system queries and actions."

        result = " ".join(summary_sentences)
        redacted_result = self.redactor.redact(result)

        # Truncate at sentence boundary if exceeding budget
        while tok.count(redacted_result) > max_tokens and len(summary_sentences) > 1:
            summary_sentences.pop()
            redacted_result = self.redactor.redact(" ".join(summary_sentences))

        return redacted_result
