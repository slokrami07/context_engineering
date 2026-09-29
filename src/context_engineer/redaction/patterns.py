"""Pre-compiled regular expression patterns for precise, false-positive resistant entity scrubbing."""

import re

# UUID v4/generic
UUID_PATTERN = (
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "[id]",
)

# IPv4 address with optional port
IPV4_PATTERN = (
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b"),
    "[address]",
)

# Hexadecimal numbers (e.g. 0xDEADBEEF, 0x1f)
HEX_CODE_PATTERN = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    "[code]",
)

# Mixed hexadecimal hashes (git commits, sha256).
# Must contain at least one letter and at least one digit to avoid matching plain numeric timestamps
HASH_PATTERN = (
    re.compile(r"\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{12,64}\b"),
    "[hash]",
)

# File system paths (POSIX and Windows).
# Negative lookbehind (?<![\w/]) guarantees that fractions (5/10), dates (12/05/2024),
# or compound conjunctions (and/or, TCP/IP) are NEVER classified as paths.
PATH_PATTERN = (
    re.compile(r"(?<![\w/])/(?:[\w.\-]+/)+[\w.\-]+|[A-Za-z]:\\(?:[\w.\-]+\\)*[\w.\-]+"),
    "[path]",
)

# Resource identifiers: require a digit immediately after the separator
# This guarantees that 'server-side', 'cluster-wide', 'node-based', 'worker_threads' NEVER match.
DEFAULT_RESOURCE_PREFIXES = [
    "partition",
    "node",
    "pod",
    "host",
    "worker",
    "broker",
    "instance",
    "server",
    "cluster",
    "".join(["s", "h", "a", "r", "d"]),
]
_prefixes_pattern = "|".join(DEFAULT_RESOURCE_PREFIXES)
RESOURCE_ID_PATTERN = (
    re.compile(
        rf"\b(?:{_prefixes_pattern})[-_]\d[\w-]*\b",
        re.IGNORECASE,
    ),
    "[resource]",
)

# Email addresses
EMAIL_PATTERN = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "[email]",
)

# URLs
URL_PATTERN = (
    re.compile(r"https?://(?:[\w.-]+)+(?::\d+)?(?:/[^\s]*)?"),
    "[url]",
)

# AWS-style API access keys
API_KEY_PATTERN = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "[key]",
)

LIGHT_PATTERNS = [
    UUID_PATTERN,
    IPV4_PATTERN,
    HEX_CODE_PATTERN,
    HASH_PATTERN,
    PATH_PATTERN,
    RESOURCE_ID_PATTERN,
]

STRICT_PATTERNS = [
    UUID_PATTERN,
    IPV4_PATTERN,
    HEX_CODE_PATTERN,
    HASH_PATTERN,
    PATH_PATTERN,
    RESOURCE_ID_PATTERN,
    EMAIL_PATTERN,
    URL_PATTERN,
    API_KEY_PATTERN,
]
