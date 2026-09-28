"""Context-Engineer: Deterministic Prefix-Cache Aware Context Management Layer.

Strictly prevents context rot and TPM/VRAM overflow while preserving prompt
prefix-caching for local LLM runtimes.
"""

from context_engineer.adapters.chatgpt import ChatGPTAdapter
from context_engineer.adapters.claude import ClaudeAdapter
from context_engineer.client import LMStudioClient
from context_engineer.config import CapPolicy, ContextConfig, default_cap_policy
from context_engineer.errors import (
    ConfigError,
    ContextBudgetError,
    ContextEngineerError,
    HistoryError,
    StructuralError,
    TokenizerError,
    TokenizerUnavailable,
)
from context_engineer.pipeline import assemble_context
from context_engineer.store.sqlite import SQLiteStore
from context_engineer.tokenizer import (
    count_message_tokens,
    count_tokens,
    count_turn_tokens,
    get_tokenizer,
)
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer
from context_engineer.types import (
    AssembledContext,
    BudgetPlan,
    IngestionPayload,
    Message,
    ToolCall,
    Turn,
    validate_history,
)

__version__ = "0.1.0"

__all__ = [
    "ContextConfig",
    "CapPolicy",
    "default_cap_policy",
    "Turn",
    "ToolCall",
    "Message",
    "BudgetPlan",
    "AssembledContext",
    "IngestionPayload",
    "validate_history",
    "count_tokens",
    "count_message_tokens",
    "count_turn_tokens",
    "get_tokenizer",
    "Tokenizer",
    "resolve_tokenizer",
    "assemble_context",
    "LMStudioClient",
    "SQLiteStore",
    "ChatGPTAdapter",
    "ClaudeAdapter",
    "ContextEngineerError",
    "ConfigError",
    "HistoryError",
    "ContextBudgetError",
    "StructuralError",
    "TokenizerError",
    "TokenizerUnavailable",
]
