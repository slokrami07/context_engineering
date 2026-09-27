"""Claude conversation export JSON adapter.

Parses Claude conversation export format into normalized Turn sequences.
"""

from typing import Any, Optional
import json

from context_engineer.types import Turn, IngestionPayload


class ClaudeAdapter:
    """Parses Claude conversation export JSON into normalized Turn objects."""

    def parse(self, raw_data: Any) -> IngestionPayload:
        """Parses Claude export data."""
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        if isinstance(raw_data, list):
            if not raw_data:
                return IngestionPayload(session_id="empty", turns=[])
            conv = raw_data[0]
        elif isinstance(raw_data, dict):
            conv = raw_data
        else:
            raise ValueError(f"Unsupported data format for Claude export: {type(raw_data)}")

        session_id = conv.get("uuid") or conv.get("id") or "claude-session"
        title = conv.get("name", "Untitled Session")
        chat_messages = conv.get("chat_messages", [])

        turns: list[Turn] = []
        turn_idx = 0

        for msg in chat_messages:
            sender = msg.get("sender", "human")
            role = "user" if sender == "human" else "assistant"
            text = msg.get("text", "")

            turn = Turn(
                id=turn_idx,
                role=role,
                content=text,
                pinned=False,
                metadata={"uuid": msg.get("uuid"), "created_at": msg.get("created_at")},
            )
            turns.append(turn)
            turn_idx += 1

        return IngestionPayload(
            session_id=session_id,
            turns=turns,
            metadata={"title": title, "source": "claude"},
        )

    def export(self, payload: IngestionPayload) -> dict[str, Any]:
        """Exports an IngestionPayload to Claude JSON format."""
        chat_messages = []
        for turn in payload.turns:
            chat_messages.append({
                "uuid": f"claude-msg-{turn.id}",
                "sender": "human" if turn.role == "user" else "assistant",
                "text": turn.get_effective_text(),
            })

        return {
            "uuid": payload.session_id,
            "name": payload.metadata.get("title", "Exported Claude Session"),
            "chat_messages": chat_messages,
        }
