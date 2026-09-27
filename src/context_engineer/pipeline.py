"""The 5-stage context assembler pipeline for context-engineer.

Strictly enforces the prefix-cacheable ribbon order:
[system] [pinned] [summary] [window turns] | [retrieved block] [question]
"""

from typing import Optional
import logging

from context_engineer.config import ContextConfig
from context_engineer.types import (
    Turn,
    Message,
    BudgetPlan,
    AssembledContext,
)
from context_engineer.tokenizer import (
    count_message_tokens,
    count_turn_tokens,
)
from context_engineer.layers.cap import apply_cap_layer
from context_engineer.layers.pin import apply_pin_layer
from context_engineer.layers.retrieve import apply_retrieve_layer
from context_engineer.layers.window import apply_window_layer
from context_engineer.layers.summarize import apply_summarize_layer

logger = logging.getLogger(__name__)


def assemble_context(
    turns: list[Turn],
    query: str,
    system_prompt: str,
    config: Optional[ContextConfig] = None,
) -> AssembledContext:
    """Assembles multi-turn context enforcing deterministic prefix-cache alignment.

    Prompt Token Ribbon Order:
    [system] [pinned] [summary] [window turns] | [retrieved block] [question]

    - Left of '|': Prefix-stable zone. Identical across queries at the same turn depth.
    - Right of '|': Dynamic suffix zone. Retrieval block and user question.

    Args:
        turns: Raw conversational history turns.
        query: Current user question or query.
        system_prompt: Top-level system instructions.
        config: Central configuration limits and thresholds.

    Returns:
        AssembledContext ready for LLM API dispatch with full budget breakdown.
    """
    if config is None:
        config = ContextConfig()

    query_tokens = count_message_tokens("user", query, model_name=config.model_name)

    # -------------------------------------------------------------
    # Stage 1: Cap Layer
    # In-place head/tail truncator for oversized tool outputs
    # -------------------------------------------------------------
    capped_turns = apply_cap_layer(turns, config=config)

    # -------------------------------------------------------------
    # Stage 2: Pin Layer
    # Deducts system prompt + pinned facts from budget ceiling
    # -------------------------------------------------------------
    pin_result = apply_pin_layer(capped_turns, system_prompt, config=config)
    pinned_turns = pin_result.pinned_turns
    unpinned_turns = pin_result.unpinned_turns
    system_tokens = pin_result.system_tokens
    pinned_tokens = pin_result.pinned_tokens

    # -------------------------------------------------------------
    # Stage 3: Retrieve Layer
    # Lookahead BM25 retrieval over candidate dropped turns
    # -------------------------------------------------------------
    # Estimate budget available for window + dynamic retrieval
    budget_for_window_and_dynamic = max(
        0,
        pin_result.remaining_budget - query_tokens - config.max_summary_tokens,
    )

    retrieve_result = apply_retrieve_layer(
        unpinned_turns=unpinned_turns,
        query=query,
        available_budget=budget_for_window_and_dynamic,
        config=config,
    )
    retrieved_turn = retrieve_result.retrieved_turn
    retrieval_tokens = retrieve_result.retrieval_tokens
    retrieved_id = retrieve_result.retrieved_turn_id

    # -------------------------------------------------------------
    # Stage 4: Window Layer
    # Recency-based contiguous turn packing
    # -------------------------------------------------------------
    excluded_ids = {retrieved_id} if retrieved_id is not None else set()
    window_budget = max(
        0,
        pin_result.remaining_budget
        - query_tokens
        - retrieval_tokens
        - config.max_summary_tokens,
    )

    window_result = apply_window_layer(
        unpinned_turns=unpinned_turns,
        available_budget=window_budget,
        config=config,
        excluded_turn_ids=excluded_ids,
    )
    window_turns = window_result.window_turns
    dropped_turns = window_result.dropped_turns
    window_tokens = window_result.window_tokens

    # -------------------------------------------------------------
    # Stage 5: Summarize Layer
    # Entity-free narrative summary of dropped non-retrieved turns
    # -------------------------------------------------------------
    summary_text, summary_tokens = apply_summarize_layer(dropped_turns, config=config)

    # -------------------------------------------------------------
    # Assemble Messages & Enforce Token Ribbon Order:
    # [system] [pinned] [summary] [window turns] | [retrieved block] [question]
    # -------------------------------------------------------------
    messages: list[Message] = []

    # 1. [system]
    messages.append(Message(role="system", content=system_prompt))

    # 2. [pinned]
    for pt in pinned_turns:
        messages.append(Message(role=pt.role, content=pt.get_effective_text()))

    # 3. [summary]
    if summary_text:
        messages.append(
            Message(
                role="system",
                content=f"[Historical Context Summary]: {summary_text}",
            )
        )

    # 4. [window turns]
    for wt in window_turns:
        messages.append(Message(role=wt.role, content=wt.get_effective_text()))

    # --- PREFIX CACHE BOUNDARY ---
    # Everything above this point is prefix-stable.
    # Everything below contains per-query dynamic variations.

    # 5. [retrieved block]
    if retrieved_turn:
        uncapped_content = retrieved_turn.get_uncapped_text()
        retrieval_msg = (
            f"[Retrieved Relevant Historical Context]:\n"
            f"- Role: {retrieved_turn.role}\n"
            f"- Turn #{retrieved_turn.id}\n"
            f"- Detail: {uncapped_content}"
        )
        messages.append(Message(role="system", content=retrieval_msg))

    # 6. [question]
    messages.append(Message(role="user", content=query))

    # Calculate final budget plan
    total_used = (
        system_tokens
        + pinned_tokens
        + summary_tokens
        + window_tokens
        + retrieval_tokens
        + query_tokens
    )
    remaining_tokens = config.effective_budget - total_used

    budget_plan = BudgetPlan(
        context_ceiling=config.context_ceiling,
        completion_reserve=config.completion_reserve,
        effective_budget=config.effective_budget,
        system_tokens=system_tokens,
        pinned_tokens=pinned_tokens,
        summary_tokens=summary_tokens,
        window_tokens=window_tokens,
        retrieval_tokens=retrieval_tokens,
        query_tokens=query_tokens,
        total_used_tokens=total_used,
        remaining_tokens=remaining_tokens,
    )

    ribbon_repr = (
        f"[sys: {system_tokens}t] "
        f"[pin: {pinned_tokens}t] "
        f"[sum: {summary_tokens}t] "
        f"[win({len(window_turns)}): {window_tokens}t] "
        f"| [ret: {retrieval_tokens}t] "
        f"[q: {query_tokens}t] "
        f"=> {total_used}/{config.effective_budget}t"
    )

    return AssembledContext(
        messages=messages,
        budget_plan=budget_plan,
        ribbon_representation=ribbon_repr,
        window_turns=window_turns,
        retrieved_turn=retrieved_turn,
        dropped_turns=dropped_turns,
        summary=summary_text,
        pinned_turns=pinned_turns,
    )
