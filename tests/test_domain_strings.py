"""Acceptance test verifying zero domain-specific strings exist in library source code (Acceptance Criterion 3)."""

import os
import re
from pathlib import Path


def test_zero_domain_strings_in_library_code() -> None:
    """Verifies that telemetry|diagnostic|shard|SRE|incident do not appear anywhere in src/context_engineer/."""
    # Matches case-insensitively
    domain_pattern = re.compile(r"\b(telemetry|diagnostic|shard|SRE|incident)\b", re.IGNORECASE)

    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src" / "context_engineer"

    assert src_dir.exists(), f"Source directory {src_dir} does not exist"

    violations: list[str] = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = Path(root) / file
            rel_path = file_path.relative_to(project_root)
            with open(file_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    match = domain_pattern.search(line)
                    if match:
                        violations.append(
                            f"{rel_path}:{line_no}: found domain keyword '{match.group(0)}' in: {line.strip()}"
                        )

    assert not violations, "Found domain-specific keywords in library code:\n" + "\n".join(
        violations
    )
