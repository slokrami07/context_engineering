"""Tests for ContextConfig validation, frozen immutability, and CapPolicy."""

import dataclasses
from typing import Any

import pytest

from context_engineer.config import ContextConfig, TurnCapRule, default_cap_policy
from context_engineer.errors import ConfigError
from context_engineer.types import Turn


def test_valid_default_config() -> None:
    cfg = ContextConfig()
    assert cfg.context_ceiling == 4096
    assert cfg.completion_reserve == 256
    assert cfg.effective_budget == 3840
    assert cfg.suffix_reserve_tokens == 768
    assert cfg.max_summary_tokens == 256


def test_config_frozen_immutability() -> None:
    cfg = ContextConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.context_ceiling = 8192  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "expected_err_pattern"),
    [
        ({"context_ceiling": 0}, "context_ceiling must be positive"),
        ({"context_ceiling": -100}, "context_ceiling must be positive"),
        ({"completion_reserve": 0}, "completion_reserve must be positive"),
        ({"completion_reserve": -50}, "completion_reserve must be positive"),
        ({"context_ceiling": 1000, "completion_reserve": 1000}, "cannot exceed or equal"),
        ({"context_ceiling": 1000, "completion_reserve": 1200}, "cannot exceed or equal"),
        ({"suffix_reserve_tokens": 0}, "suffix_reserve_tokens must be positive"),
        ({"suffix_reserve_tokens": -10}, "suffix_reserve_tokens must be positive"),
        ({"max_summary_tokens": 0}, "max_summary_tokens must be positive"),
        ({"retrieval_top_k": 0}, "retrieval_top_k must be at least 1"),
        ({"min_recent_turns": 0}, "min_recent_turns must be at least 1"),
        ({"budget_safety_margin": -0.1}, "budget_safety_margin must be in range"),
        ({"budget_safety_margin": 1.0}, "budget_safety_margin must be in range"),
        ({"redaction_mode": "invalid_mode"}, "Invalid redaction_mode"),
        ({"summary_placement": "invalid_placement"}, "Invalid summary_placement"),
        ({"cap_threshold": 0}, "cap_threshold must be positive"),
        ({"cap_head_tokens": 0}, "must be >= 1"),
        ({"cap_tail_tokens": 0}, "must be >= 1"),
        (
            {"cap_threshold": 30, "cap_head_tokens": 20, "cap_tail_tokens": 15},
            "cannot exceed cap_threshold",
        ),
        (
            {
                "context_ceiling": 1000,
                "completion_reserve": 200,
                "suffix_reserve_tokens": 500,
                "max_summary_tokens": 400,
            },
            "Fixed reserves",
        ),
    ],
)
def test_invalid_config_raises_config_error(
    kwargs: dict[str, Any], expected_err_pattern: str
) -> None:
    with pytest.raises(ConfigError, match=expected_err_pattern):
        ContextConfig(**kwargs)


def test_turn_cap_rule_validation() -> None:
    with pytest.raises(ConfigError, match="threshold must be positive"):
        TurnCapRule(threshold=0)
    with pytest.raises(ConfigError, match="head_tokens must be >= 1"):
        TurnCapRule(head_tokens=0)
    with pytest.raises(ConfigError, match="tail_tokens must be >= 1"):
        TurnCapRule(tail_tokens=0)
    with pytest.raises(ConfigError, match="cannot exceed threshold"):
        TurnCapRule(threshold=100, head_tokens=70, tail_tokens=40)


def test_cap_policy_matching() -> None:
    policy = default_cap_policy()
    # Pinned turns are never capped
    pinned_turn = Turn(id=0, role="assistant", content="Pinned fact", pinned=True)
    assert policy.for_turn(pinned_turn) is None

    # Normal user message not capped by default
    user_turn = Turn(id=1, role="user", content="Hello")
    assert policy.for_turn(user_turn) is None

    # Tool turn is capped
    tool_turn = Turn(id=2, role="tool", content="Output from tool", kind="tool_result")
    rule = policy.for_turn(tool_turn)
    assert rule is not None
    assert rule.threshold == 400
