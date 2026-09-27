"""Unit tests for ChatGPT and Claude conversation export adapters."""

import pytest
import json

from context_engineer.adapters.chatgpt import ChatGPTAdapter
from context_engineer.adapters.claude import ClaudeAdapter
from context_engineer.types import Turn, IngestionPayload


def test_chatgpt_adapter_parsing() -> None:
    sample_chatgpt_data = {
        "id": "chatcmpl-test-123",
        "title": "Incident Triage",
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["node1"], "message": None},
            "node1": {
                "id": "node1",
                "parent": "root",
                "children": ["node2"],
                "message": {
                    "id": "m1",
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": ["You are a triage bot."]},
                    "create_time": 1000.0,
                },
            },
            "node2": {
                "id": "node2",
                "parent": "node1",
                "children": ["node3"],
                "message": {
                    "id": "m2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["What happened to partition 19?"]},
                    "create_time": 1001.0,
                },
            },
            "node3": {
                "id": "node3",
                "parent": "node2",
                "children": [],
                "message": {
                    "id": "m3",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Partition 19 failed with error."]},
                    "create_time": 1002.0,
                },
            },
        },
        "current_node": "node3",
    }

    adapter = ChatGPTAdapter()
    payload = adapter.parse(sample_chatgpt_data)

    assert payload.session_id == "chatcmpl-test-123"
    assert payload.system_prompt == "You are a triage bot."
    assert len(payload.turns) == 2
    assert payload.turns[0].role == "user"
    assert "partition 19" in payload.turns[0].content
    assert payload.turns[1].role == "assistant"

    # Export round-trip
    exported = adapter.export(payload)
    assert exported["id"] == "chatcmpl-test-123"
    assert "mapping" in exported


def test_claude_adapter_parsing() -> None:
    sample_claude_data = {
        "uuid": "claude-session-999",
        "name": "Cluster Investigation",
        "chat_messages": [
            {"uuid": "cm1", "sender": "human", "text": "Please check shard-19."},
            {"uuid": "cm2", "sender": "assistant", "text": "Shard-19 write buffer was saturated."},
        ],
    }

    adapter = ClaudeAdapter()
    payload = adapter.parse(sample_claude_data)

    assert payload.session_id == "claude-session-999"
    assert len(payload.turns) == 2
    assert payload.turns[0].role == "user"
    assert payload.turns[0].content == "Please check shard-19."
    assert payload.turns[1].role == "assistant"

    # Export round-trip
    exported = adapter.export(payload)
    assert exported["uuid"] == "claude-session-999"
    assert len(exported["chat_messages"]) == 2
