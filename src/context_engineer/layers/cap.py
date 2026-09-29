"""Layer 1: Pure, idempotent head/tail message and tool output truncator (Algorithm A10).

Caps oversized tool outputs or user messages down to configured thresholds using
domain-agnostic strategies (JSON structure preservation, line elision, or token slicing).
Preserves raw content for uncapped retrieval, and guarantees cap(cap(t)) == cap(t).
"""

import dataclasses
import json
from typing import Any

from context_engineer.config import ContextConfig, TurnCapRule
from context_engineer.tokenizers import Tokenizer, resolve_tokenizer, slice_head_tail
from context_engineer.types import Turn


def _cap_json(text: str, rule: TurnCapRule, tok: Tokenizer) -> str | None:
    """Attempts structured JSON elision while preserving top-level keys and valid JSON format."""
    try:
        data = json.loads(text)
    except Exception:
        return None

    if not isinstance(data, (dict, list)):
        return None

    if isinstance(data, dict):
        keys = list(data.keys())
        for max_keys in range(len(keys), 0, -1):
            truncated: dict[str, Any] = {}
            for k in keys[:max_keys]:
                v = data[k]
                if isinstance(v, list) and len(v) > 2:
                    truncated[k] = v[:2] + [f"... ({len(v) - 2} items elided) ..."]
                elif isinstance(v, dict) and len(v) > 2:
                    sub_keys = list(v.keys())[:2]
                    truncated[k] = {sk: v[sk] for sk in sub_keys}
                    truncated[k]["_elided"] = f"... ({len(v) - 2} keys elided) ..."
                else:
                    truncated[k] = v
            if max_keys < len(keys):
                truncated["_elided"] = f"... ({len(keys) - max_keys} keys elided) ..."
            out = json.dumps(truncated)
            if tok.count(out) <= rule.threshold:
                return out

        minimal_dict = {"_elided": f"... ({len(keys)} keys elided) ..."}
        out = json.dumps(minimal_dict)
        if tok.count(out) <= rule.threshold:
            return out

    elif isinstance(data, list):
        for max_items in range(len(data), 0, -1):
            truncated_list = data[:max_items] + [f"... ({len(data) - max_items} items elided) ..."]
            out = json.dumps(truncated_list)
            if tok.count(out) <= rule.threshold:
                return out

        minimal_list = [f"... ({len(data)} items elided) ..."]
        out = json.dumps(minimal_list)
        if tok.count(out) <= rule.threshold:
            return out

    return None


def _cap_lines(text: str, rule: TurnCapRule, tok: Tokenizer) -> str | None:
    """Attempts line-based elision preserving head and tail lines."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= 6:
        return None

    # Estimate line budget
    h_count = max(2, len(lines) // 4)
    t_count = max(2, len(lines) // 6)
    head_lines = lines[:h_count]
    tail_lines = lines[-t_count:]
    elided = len(lines) - (h_count + t_count)

    if elided > 0:
        marker = f"\n[... {elided} lines elided ...]\n"
        out = "".join(head_lines) + marker + "".join(tail_lines)
        if tok.count(out) <= rule.threshold:
            return out

    return None


def _cap_text(text: str, rule: TurnCapRule, tok: Tokenizer) -> str:
    """Caps text according to the selected strategy, falling back safely to token_slice."""
    if tok.count(text) <= rule.threshold:
        return text

    if rule.strategy == "json":
        json_out = _cap_json(text, rule, tok)
        if json_out is not None:
            return json_out

    if rule.strategy in ("json", "lines"):
        lines_out = _cap_lines(text, rule, tok)
        if lines_out is not None:
            return lines_out

    # Fallback / standard token_slice strategy
    return slice_head_tail(
        text,
        head_tokens=rule.head_tokens,
        tail_tokens=rule.tail_tokens,
        tokenizer=tok,
    )


def cap_turn(
    turn: Turn,
    config: ContextConfig,
    tokenizer: Tokenizer | None = None,
) -> Turn:
    """Caps oversized tool_output or content in a single Turn (Algorithm A10).

    Idempotent: cap(cap(t)) == cap(t). Never caps pinned turns.
    Preserves raw_content and raw_tool_output intact for later retrieval.
    """
    if turn.pinned or turn.is_capped:
        return turn

    rule = config.cap_policy.for_turn(turn)
    if rule is None:
        return turn

    tok = tokenizer or resolve_tokenizer(config.tokenizer)

    tool_output = turn.tool_output
    raw_tool_output = turn.raw_tool_output
    content = turn.content
    raw_content = turn.raw_content
    modified = False

    # 1. Cap tool_output if present
    if tool_output and tok.count(tool_output) > rule.threshold:
        if raw_tool_output is None:
            raw_tool_output = tool_output
        tool_output = _cap_text(tool_output, rule, tok)
        modified = True

    # 2. Cap content if it represents tool results or user message permitted by policy
    should_cap_content = (
        turn.kind == "tool_result"
        or turn.role == "tool"
        or (turn.role == "user" and config.cap_policy.cap_user_messages)
    )

    if should_cap_content and tok.count(content) > rule.threshold:
        if raw_content is None:
            raw_content = content
        content = _cap_text(content, rule, tok)
        modified = True

    if not modified:
        return turn

    return dataclasses.replace(
        turn,
        content=content,
        tool_output=tool_output,
        raw_content=raw_content,
        raw_tool_output=raw_tool_output,
        is_capped=True,
    )


def apply_cap_layer(
    turns: list[Turn],
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Turn]:
    """Layer 1 pipeline entry point: caps oversized turn contents across all unpinned turns.

    Args:
        turns: List of conversational turns.
        config: Central configuration instance (uses default if None).
        tokenizer: Tokenizer instance (resolved from config if None).

    Returns:
        List of turns with capped contents, retaining original objects if unchanged.
    """
    if config is None:
        config = ContextConfig()

    tok = tokenizer or resolve_tokenizer(config.tokenizer)
    return [cap_turn(t, config, tok) for t in turns]
