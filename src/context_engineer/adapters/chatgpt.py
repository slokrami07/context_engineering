"""ChatGPT conversation export JSON adapter.

Parses standard ChatGPT conversations.json tree mappings into normalized Turn sequences.
"""

from typing import Any, Optional
import json

from context_engineer.types import Turn, IngestionPayload


class ChatGPTAdapter:
    """Parses ChatGPT export JSON into a sequence of normalized Turn objects."""

    def parse(self, raw_data: Any) -> IngestionPayload:
        """Parses ChatGPT export data.

        Accepts either a JSON string, a single conversation dict, or a list of conversations.
        """
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        if isinstance(raw_data, list):
            # Pick first conversation if a full export list was passed
            if not raw_data:
                return IngestionPayload(session_id="empty", turns=[])
            conv = raw_data[0]
        elif isinstance(raw_data, dict):
            conv = raw_data
        else:
            raise ValueError(f"Unsupported data format for ChatGPT export: {type(raw_data)}")

        session_id = conv.get("id") or conv.get("conversation_id") or "chatgpt-session"
        title = conv.get("title", "Untitled Session")
        mapping = conv.get("mapping", {})

        turns: list[Turn] = []
        system_prompt: Optional[str] = None

        # Build chronological thread from tree mapping
        # Find root node (node without parent or parent not in mapping)
        node_map = {}
        children_map = {}
        for node_id, node in mapping.items():
            node_map[node_id] = node
            parent_id = node.get("parent")
            if parent_id:
                children_map.setdefault(parent_id, []).append(node_id)

        # Find current node or follow standard traversal
        current_node_id = conv.get("current_node")
        ordered_node_ids = []

        if current_node_id and current_node_id in node_map:
            # Trace backwards from current_node to root
            curr = current_node_id
            while curr and curr in node_map:
                ordered_node_ids.append(curr)
                curr = node_map[curr].get("parent")
            ordered_node_ids.reverse()
        else:
            # Fallback: topological or creation order
            nodes = [n for n in mapping.values() if n.get("message")]
            nodes.sort(key=lambda x: (x.get("message", {}).get("create_time") or 0))
            ordered_node_ids = [n["id"] for n in nodes]

        turn_idx = 0
        for nid in ordered_node_ids:
            node = node_map.get(nid)
            if not node:
                continue
            msg = node.get("message")
            if not msg:
                continue

            author = msg.get("author", {})
            role = author.get("role", "user")
            content_dict = msg.get("content", {})
            parts = content_dict.get("parts", [])

            # Join parts
            content_text = ""
            for part in parts:
                if isinstance(part, str):
                    content_text += part
                elif isinstance(part, dict) and "text" in part:
                    content_text += part["text"]

            if not content_text and role not in ("tool",):
                continue

            # Check if this is top-level system message
            if role == "system" and system_prompt is None and turn_idx == 0:
                system_prompt = content_text
                continue

            tool_output = None
            if role == "tool" or author.get("name") in ("browser", "python", "dalle"):
                tool_output = content_text
                content_text = f"Tool execution output ({author.get('name', 'tool')}):"

            turn = Turn(
                id=turn_idx,
                role=role,
                content=content_text,
                tool_output=tool_output,
                pinned=False,
                metadata={"message_id": msg.get("id"), "create_time": msg.get("create_time")},
            )
            turns.append(turn)
            turn_idx += 1

        return IngestionPayload(
            session_id=session_id,
            turns=turns,
            system_prompt=system_prompt,
            metadata={"title": title, "source": "chatgpt"},
        )

    def export(self, payload: IngestionPayload) -> dict[str, Any]:
        """Exports an IngestionPayload back to a minimal ChatGPT JSON structure."""
        mapping: dict[str, Any] = {}
        last_id = "root"
        mapping["root"] = {"id": "root", "parent": None, "children": [], "message": None}

        for turn in payload.turns:
            node_id = f"node_{turn.id}"
            mapping[last_id]["children"].append(node_id)
            parts = [turn.content]
            if turn.tool_output:
                parts.append(f"\n[Tool Output]: {turn.tool_output}")

            mapping[node_id] = {
                "id": node_id,
                "parent": last_id,
                "children": [],
                "message": {
                    "id": f"msg_{turn.id}",
                    "author": {"role": turn.role},
                    "content": {"content_type": "text", "parts": parts},
                },
            }
            last_id = node_id

        return {
            "id": payload.session_id,
            "title": payload.metadata.get("title", "Exported Session"),
            "current_node": last_id,
            "mapping": mapping,
        }
