"""Golden unit tests for entity scrubbing patterns and false-positive guards (Algorithm A7)."""

import pytest

from context_engineer.redaction.redactor import RegexRedactor


@pytest.fixture
def redactor_light() -> RegexRedactor:
    return RegexRedactor(mode="light")


@pytest.fixture
def redactor_strict() -> RegexRedactor:
    return RegexRedactor(mode="strict")


def test_false_positive_guards_paths(redactor_light: RegexRedactor) -> None:
    """Verifies that negative lookbehinds guard conjunctions, dates, and ratios from being scrubbed as paths."""
    text = "Choose and/or options for TCP/IP protocol on 12/05/2024 with 5/10 ratio."
    scrubbed = redactor_light.redact(text)

    assert "and/or" in scrubbed
    assert "TCP/IP" in scrubbed
    assert "12/05/2024" in scrubbed
    assert "5/10" in scrubbed
    assert "[path]" not in scrubbed


def test_false_positive_guards_resources(redactor_light: RegexRedactor) -> None:
    """Verifies that requiring digits prevents generic hyphenated terms from being scrubbed as resources."""
    text = (
        "Implementing server-side validation and cluster-wide synchronization "
        "with node-based execution and worker_threads pool."
    )
    scrubbed = redactor_light.redact(text)

    assert "server-side" in scrubbed
    assert "cluster-wide" in scrubbed
    assert "node-based" in scrubbed
    assert "worker_threads" in scrubbed
    assert "[resource]" not in scrubbed


def test_false_positive_guards_hashes(redactor_light: RegexRedactor) -> None:
    """Verifies that numeric timestamps are not scrubbed as git/sha commit hashes."""
    text = "The timestamp was 1699999999999 recorded in ms."
    scrubbed = redactor_light.redact(text)

    assert "1699999999999" in scrubbed
    assert "[hash]" not in scrubbed


def test_true_positive_scrubbing_light(redactor_light: RegexRedactor) -> None:
    """Verifies true-positive redaction for UUIDs, IPs, hex numbers, commit hashes, paths, and resource IDs."""
    text = (
        "Node worker-12 crashed with error 0xDEADBEEF at 10.0.0.1:8080. "
        "Commit deadbeefcafe1234 referenced log /var/log/app.log and C:\\Temp\\a.log. "
        "Job uuid 123e4567-e89b-12d3-a456-426614174000."
    )
    scrubbed = redactor_light.redact(text)

    assert "worker-12" not in scrubbed
    assert "[resource]" in scrubbed

    assert "0xDEADBEEF" not in scrubbed
    assert "[code]" in scrubbed

    assert "10.0.0.1:8080" not in scrubbed
    assert "[address]" in scrubbed

    assert "deadbeefcafe1234" not in scrubbed
    assert "[hash]" in scrubbed

    assert "/var/log/app.log" not in scrubbed
    assert "C:\\Temp\\a.log" not in scrubbed
    assert "[path]" in scrubbed

    assert "123e4567-e89b-12d3-a456-426614174000" not in scrubbed
    assert "[id]" in scrubbed


def test_true_positive_scrubbing_strict(redactor_strict: RegexRedactor) -> None:
    """Verifies strict mode redaction of emails, URLs, and API keys."""
    text = (
        "Contact security@company.org or visit https://vault.internal/secrets. "
        "Access key AKIAIOSFODNN7EXAMPLE was reported."
    )
    scrubbed = redactor_strict.redact(text)

    assert "security@company.org" not in scrubbed
    assert "[email]" in scrubbed

    assert "https://vault.internal/secrets" not in scrubbed
    assert "[url]" in scrubbed

    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
    assert "[key]" in scrubbed


def test_custom_resource_prefixes() -> None:
    """Verifies dynamic registration of custom resource prefixes."""
    redactor = RegexRedactor(mode="light", resource_prefixes=["redis", "db"])
    text = "Failover occurred on redis-01 and db-3 while cache remained."
    scrubbed = redactor.redact(text)

    assert "redis-01" not in scrubbed
    assert "db-3" not in scrubbed
    assert "[resource]" in scrubbed
