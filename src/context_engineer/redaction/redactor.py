"""Configurable entity redactor implementing the Redactor protocol."""

import re
from typing import Literal

from context_engineer.protocols import Redactor
from context_engineer.redaction.patterns import LIGHT_PATTERNS, STRICT_PATTERNS


class RegexRedactor(Redactor):
    """Deterministic regex redactor scrubbing identifiers to prevent stale fact hallucination."""

    def __init__(
        self,
        mode: Literal["none", "light", "strict"] = "light",
        extra_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        resource_prefixes: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.patterns: list[tuple[re.Pattern[str], str]] = []

        if mode == "light":
            self.patterns.extend(LIGHT_PATTERNS)
        elif mode == "strict":
            self.patterns.extend(STRICT_PATTERNS)

        if resource_prefixes:
            prefix_alt = "|".join(re.escape(p) for p in resource_prefixes)
            custom_resource = (
                re.compile(rf"\b(?:{prefix_alt})[-_]\d[\w-]*\b", re.IGNORECASE),
                "[resource]",
            )
            self.patterns.append(custom_resource)

        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def redact(self, text: str) -> str:
        """Applies configured scrubbing patterns in single pass."""
        if self.mode == "none" or not text:
            return text

        scrubbed = text
        for pattern, replacement in self.patterns:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed
