"""Context-Engineer: Deterministic Prefix-Cache Aware Context Management Layer.

Strictly prevents context rot and TPM/VRAM overflow while preserving prompt
prefix-caching for local LLM runtimes.
"""

from context_engineer.config import ContextConfig
from context_engineer.types import (
    Turn,
    Message,
    BudgetPlan,
    AssembledContext,
    IngestionPayload,
)
from context_engineer.tokenizer import (
    count_tokens,
    count_message_tokens,
    count_turn_tokens,
    get_tokenizer,
)
from context_engineer.pipeline import assemble_context
from context_engineer.client import LMStudioClient
from context_engineer.store.sqlite import SQLiteStore
from context_engineer.adapters.chatgpt import ChatGPTAdapter
from context_engineer.adapters.claude import ClaudeAdapter

__version__ = "0.1.0"

__all__ = [
    "ContextConfig",
    "Turn",
    "Message",
    "BudgetPlan",
    "AssembledContext",
    "IngestionPayload",
    "count_tokens",
    "count_message_tokens",
    "count_turn_tokens",
    "get_tokenizer",
    "assemble_context",
    "LMStudioClient",
    "SQLiteStore",
    "ChatGPTAdapter",
    "ClaudeAdapter",
]
