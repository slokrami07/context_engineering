"""Core data structures, types, and history validators for context-engineer."""

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from context_engineer.errors import HistoryError


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represents a structured function or tool invocation by an LLM assistant."""

    id: str
    name: str
    arguments: str  # JSON-encoded arguments string


@dataclass(frozen=True, slots=True)
class Turn:
    """Represents an immutable conversational turn in a multi-turn session.

    Attributes:
        id: Unique monotonic sequence identifier.
        role: Participant role ('system', 'user', 'assistant', 'tool').
        content: Text content of the turn (may be capped by Layer 1).
        kind: Structural turn discriminator ('message', 'tool_call', 'tool_result').
        tool_calls: Structured tool call invocations (when kind == 'tool_call').
        tool_call_id: Target tool call identifier for tool results (when kind == 'tool_result').
        name: Function or author name.
        tool_output: DEPRECATED legacy output string (kept for backward compatibility).
        pinned: Invariant fact flag; pinned turns are never evicted and placed upfront.
        raw_content: Original uncapped content before Layer 1 truncation.
        raw_tool_output: Original uncapped tool output before Layer 1 truncation.
        is_capped: Flag indicating whether this turn was truncated by Layer 1.
        metadata: Defensive mapping of arbitrary user/system metadata.
    """

    id: int
    role: str
    content: str
    kind: Literal["message", "tool_call", "tool_result"] = "message"
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    tool_output: str | None = None
    pinned: bool = False
    raw_content: str | None = None
    raw_tool_output: str | None = None
    is_capped: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Emit deprecation warning if legacy tool_output is used directly
        if self.tool_output is not None:
            warnings.warn(
                "Turn.tool_output is deprecated and will be removed in v0.3.0. "
                "Use kind='tool_result' and content with tool_call_id instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Defensively copy metadata to prevent mutable aliasing across copies (T3)
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        """Computes a deterministic, collision-resistant hash of this turn's semantic content."""
        hasher = hashlib.sha256()
        hasher.update(str(self.id).encode("utf-8"))
        hasher.update(self.role.encode("utf-8"))
        hasher.update(self.kind.encode("utf-8"))
        hasher.update(self.content.encode("utf-8"))
        if self.tool_output:
            hasher.update(self.tool_output.encode("utf-8"))
        if self.tool_call_id:
            hasher.update(self.tool_call_id.encode("utf-8"))
        for tc in self.tool_calls:
            hasher.update(tc.id.encode("utf-8"))
            hasher.update(tc.name.encode("utf-8"))
            hasher.update(tc.arguments.encode("utf-8"))
        return hasher.hexdigest()

    def get_effective_text(self) -> str:
        """Returns the full active text for this turn (content + legacy tool_output)."""
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
            "kind": self.kind,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls
            ],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "tool_output": self.tool_output,
            "pinned": self.pinned,
            "raw_content": self.raw_content,
            "raw_tool_output": self.raw_tool_output,
            "is_capped": self.is_capped,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        """Reconstructs a Turn from a dictionary with backward compatibility."""
        tool_calls_raw = data.get("tool_calls", [])
        tool_calls = tuple(
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", "{}"),
            )
            for tc in tool_calls_raw
        )

        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            kind=data.get("kind", "message"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            tool_output=data.get("tool_output"),
            pinned=data.get("pinned", False),
            raw_content=data.get("raw_content"),
            raw_tool_output=data.get("raw_tool_output"),
            is_capped=data.get("is_capped", False),
            metadata=dict(data.get("metadata", {})),
        )


def validate_history(history: Sequence[Turn]) -> None:
    """Validates structural invariants of a conversational history sequence.

    Enforces:
    1. IDs are unique and strictly monotonically increasing (0, 1, 2, ...).
    2. Tool results follow their corresponding tool calls and do not exist as orphans.

    Raises:
        HistoryError: If any sequence constraint is violated.
    """
    if not history:
        return

    seen_ids: set[int] = set()
    prev_id = -1
    pending_tool_call_ids: set[str] = set()

    for idx, turn in enumerate(history):
        # 1. Monotonicity and uniqueness
        if turn.id in seen_ids:
            raise HistoryError(
                f"History invariant violated at index {idx}: duplicate turn id {turn.id} detected."
            )
        if idx > 0 and turn.id <= prev_id:
            raise HistoryError(
                f"History invariant violated at index {idx}: non-monotonic turn id "
                f"(current id: {turn.id}, previous id: {prev_id}). IDs must strictly increase."
            )
        seen_ids.add(turn.id)
        prev_id = turn.id

        # 2. Tool call / result tracking
        if turn.kind == "tool_call":
            for tc in turn.tool_calls:
                pending_tool_call_ids.add(tc.id)
        elif (
            turn.kind == "tool_result"
            and turn.tool_call_id is not None
            and turn.tool_call_id in pending_tool_call_ids
        ):
            pending_tool_call_ids.remove(turn.tool_call_id)


@dataclass(frozen=True, slots=True)
class Message:
    """Standard OpenAI/ChatML compatible message payload.

    Attributes:
        role: 'system', 'user', 'assistant', or 'tool'.
        content: The message text payload.
        name: Optional function or author name.
        tool_call_id: Tool call identifier for tool results.
        tool_calls: Tool call objects emitted by assistant turns.
    """

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Message to an OpenAI/ChatML compatible payload dictionary."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return payload


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """Token budget accounting across all context layers.

    Reports both estimated per-zone token allocation and measured final render tokens.
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
    measured_total_tokens: int | None = None

    @property
    def is_valid(self) -> bool:
        """True if total token usage stays strictly within the effective budget ceiling."""
        check_tokens = (
            self.measured_total_tokens
            if self.measured_total_tokens is not None
            else self.total_used_tokens
        )
        return check_tokens <= self.effective_budget


@dataclass(frozen=True, slots=True)
class AssemblyReport:
    """Telemetry report emitted with every context assembly for observability and verification."""

    tokenizer_id: str
    estimated_total_tokens: int
    measured_total_tokens: int | None
    window_start_index: int
    boundary_changed: bool
    compaction_occurred: bool
    dropped_turn_ids: tuple[int, ...]
    retrieved_turn_ids: tuple[int, ...]
    summary_hash: str | None
    prefix_hash: str
    predicted_cache_stable_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "estimated_total_tokens": self.estimated_total_tokens,
            "measured_total_tokens": self.measured_total_tokens,
            "window_start_index": self.window_start_index,
            "boundary_changed": self.boundary_changed,
            "compaction_occurred": self.compaction_occurred,
            "dropped_turn_ids": list(self.dropped_turn_ids),
            "retrieved_turn_ids": list(self.retrieved_turn_ids),
            "summary_hash": self.summary_hash,
            "prefix_hash": self.prefix_hash,
            "predicted_cache_stable_tokens": self.predicted_cache_stable_tokens,
        }


@dataclass
class AssembledContext:
    """Complete assembled prompt context ready for dispatch to LLM runtimes."""

    messages: list[Message]
    budget_plan: BudgetPlan
    ribbon_representation: str
    window_turns: list[Turn]
    retrieved_turns: tuple[Turn, ...] = ()
    retrieved_turn: Turn | None = None  # Backward-compatible property / field
    dropped_turns: list[Turn] = field(default_factory=list)
    summary: str | None = None
    pinned_turns: list[Turn] = field(default_factory=list)
    report: AssemblyReport | None = None

    def __post_init__(self) -> None:
        # Keep retrieved_turn synchronized with retrieved_turns
        if self.retrieved_turn is None and self.retrieved_turns:
            self.retrieved_turn = self.retrieved_turns[0]
        elif self.retrieved_turn is not None and not self.retrieved_turns:
            self.retrieved_turns = (self.retrieved_turn,)


@dataclass
class IngestionPayload:
    """Normalized multi-turn conversation payload produced by export adapters."""

    session_id: str
    turns: list[Turn]
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
