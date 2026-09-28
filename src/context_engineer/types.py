"""Core data structures and types for context-engineer."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """Represents a single conversational turn in the multi-turn session.

    Attributes:
        id: Unique monotonic sequence identifier for the turn.
        role: Participant role ('system', 'user', 'assistant', 'tool').
        content: Text content of the turn (may be capped by Layer 1).
        tool_output: Optional tool/function output associated with this turn.
        pinned: Invariant fact flag; pinned turns are never evicted and placed upfront.
        raw_content: Original uncapped content before Layer 1 truncation.
        raw_tool_output: Original uncapped tool output before Layer 1 truncation.
        is_capped: Boolean flag indicating whether this turn was truncated by Layer 1.
        metadata: Arbitrary user/system metadata dictionary.
    """

    id: int
    role: str
    content: str
    tool_output: str | None = None
    pinned: bool = False
    raw_content: str | None = None
    raw_tool_output: str | None = None
    is_capped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.raw_content is None:
            self.raw_content = self.content
        if self.raw_tool_output is None and self.tool_output is not None:
            self.raw_tool_output = self.tool_output

    def get_effective_text(self) -> str:
        """Returns the full active text for this turn (content + tool_output)."""
        if self.tool_output:
            return f"{self.content}\n[Tool Output]: {self.tool_output}"
        return self.content

    def get_uncapped_text(self) -> str:
        """Returns the full uncapped text for this turn."""
        content = self.raw_content if self.raw_content is not None else self.content
        tool = self.raw_tool_output if self.raw_tool_output is not None else self.tool_output
        if tool:
            return f"{content}\n[Tool Output]: {tool}"
        return content

    def to_dict(self) -> dict[str, Any]:
        """Serializes the turn to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tool_output": self.tool_output,
            "pinned": self.pinned,
            "raw_content": self.raw_content,
            "raw_tool_output": self.raw_tool_output,
            "is_capped": self.is_capped,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        """Reconstructs a Turn from a dictionary."""
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            tool_output=data.get("tool_output"),
            pinned=data.get("pinned", False),
            raw_content=data.get("raw_content"),
            raw_tool_output=data.get("raw_tool_output"),
            is_capped=data.get("is_capped", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Message:
    """Standard OpenAI/ChatML compatible message payload.

    Attributes:
        role: 'system', 'user', 'assistant', or 'tool'.
        content: The message text payload.
        name: Optional function or author name.
    """

    role: str
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class BudgetPlan:
    """Token budget accounting across all context layers.

    Guarantees that total_used_tokens <= effective_budget, strictly preventing
    context rot and TPM/VRAM overflow.
    """

    context_ceiling: int
    completion_reserve: int
    effective_budget: int
    system_tokens: int
    pinned_tokens: int
    summary_tokens: int
    window_tokens: int
    retrieval_tokens: int
    query_tokens: int
    total_used_tokens: int
    remaining_tokens: int

    @property
    def is_valid(self) -> bool:
        """True if token usage stays within the effective prompt budget."""
        return self.total_used_tokens <= self.effective_budget


@dataclass
class AssembledContext:
    """Complete assembled prompt context ready for dispatch to the LLM runtime."""

    messages: list[Message]
    budget_plan: BudgetPlan
    ribbon_representation: str
    window_turns: list[Turn]
    retrieved_turn: Turn | None = None
    dropped_turns: list[Turn] = field(default_factory=list)
    summary: str | None = None
    pinned_turns: list[Turn] = field(default_factory=list)


@dataclass
class IngestionPayload:
    """Normalized multi-turn conversation payload produced by export adapters."""

    session_id: str
    turns: list[Turn]
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
