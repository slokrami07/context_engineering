"""Domain exceptions and error hierarchy for context-engineer."""


class ContextEngineerError(Exception):
    """Base exception for all errors raised by context-engineer."""


class ConfigError(ContextEngineerError, ValueError):
    """Raised when ContextConfig fails validation."""


class HistoryError(ContextEngineerError, ValueError):
    """Raised when conversation history violates structural or sequence rules (e.g. non-monotonic IDs, orphan tool results)."""


class ContextBudgetError(ContextEngineerError, RuntimeError):
    """Raised when context reserves or rendered prompt exceed the configured ceiling."""


class StructuralError(ContextEngineerError, ValueError):
    """Raised when rendered messages violate role alternation or tool call/result pairing."""


class TokenizerError(ContextEngineerError, RuntimeError):
    """Raised when tokenizer loading or execution fails and strict mode is active."""


class TokenizerUnavailable(ContextEngineerError, ImportError):
    """Raised when an optional tokenizer backend is not installed or available locally."""
