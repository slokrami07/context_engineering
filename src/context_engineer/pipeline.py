"""The 5-stage context assembler pipeline for context-engineer (Algorithm A3).

Enforces prefix stability, fixed reserves, and strict budget containment:
[system + pinned + summary] [window turns] | [retrieved block + question]
<------------- prefix-stable zone -------->  <------- dynamic suffix ------>
"""

import hashlib
import logging
from collections.abc import Sequence

from context_engineer.config import ContextConfig
from context_engineer.errors import ContextBudgetError
from context_engineer.layers.cap import apply_cap_layer
from context_engineer.layers.pin import split_pinned
from context_engineer.layers.retrieve import BM25Retriever
from context_engineer.layers.window import memoized_turn_cost, window_boundary
from context_engineer.redaction.redactor import RegexRedactor
from context_engineer.render import gate, render, validate_messages
from context_engineer.summarizers.extractive import ExtractiveSummarizer
from context_engineer.tokenizers import (
    Tokenizer,
    resolve_tokenizer,
    truncate_text_to_tokens,
)
from context_engineer.types import (
    AssembledContext,
    AssemblyReport,
    BudgetPlan,
    Message,
    Turn,
    validate_history,
)

logger = logging.getLogger(__name__)


def assemble_context(
    turns: Sequence[Turn],
    query: str,
    system_prompt: str = "",
    config: ContextConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> AssembledContext:
    """Assembles multi-turn conversational context enforcing deterministic prefix-cache alignment (Algorithm A3).

    Pipeline stages:
    1. Validation: Verifies chronological IDs and call/result sequencing (T4).
    2. Pin Partition: Separates pinned invariant facts (pinned turns never capped, I8).
    3. Cap Layer: Idempotently caps oversized tool results and allowed messages (A10).
    4. Fixed Reserves: Computes window budget independently of query or retrieval size (A1, I2, I3).
    5. Window Boundary: Computes stateless quantized boundary on coarse grid (A2, I1, I4).
    6. Summarize: Extractive summary purely derived from dropped turns (A6, S1-S8).
    7. Retrieval: Hard-bounded BM25 retrieval over dropped turns (A9, R1-R3).
    8. Render & Gate: Unified message rendering and final template gate measurement (A5, I5, I6).

    Args:
        turns: Conversational history turns.
        query: Current user question / query.
        system_prompt: Invariant system prompt.
        config: Central configuration limits and thresholds.
        tokenizer: Optional tokenizer instance.

    Returns:
        AssembledContext with unified messages, budget plan, and assembly report.

    Raises:
        ContextBudgetError: If fixed reserves or rendered prompt exceed ceiling (I5, I9).
        HistoryError: If input turns violate chronological or structural ordering (T4).
        StructuralError: If rendered messages violate template requirements (I6).
    """
    cfg = config or ContextConfig()
    tok = tokenizer or resolve_tokenizer(cfg.tokenizer, strict=cfg.strict_tokenizer)

    # 1. Structural History Validation (T4)
    validate_history(turns)

    # 2. Pin Partitioning (I8: pinned turns are invariant and NEVER capped)
    pinned, unpinned = split_pinned(turns)

    # 3. Cap Layer (A10: pure, idempotent, memoized)
    unpinned = apply_cap_layer(unpinned, config=cfg, tokenizer=tok)

    # 4. Fixed Reserves (Algorithm A1: query-independent window budget)
    effective_ceiling = cfg.context_ceiling - cfg.completion_reserve

    cost_system = 0
    if system_prompt:
        cost_system = tok.count_messages(
            [Message("system", system_prompt)], add_generation_prompt=False
        )

    cost_pinned = 0
    if pinned:
        pinned_facts = "\n".join(f"- {p.get_effective_text()}" for p in pinned)
        cost_pinned = tok.count_messages(
            [Message("system", f"Pinned facts:\n{pinned_facts}")],
            add_generation_prompt=False,
        )

    fixed_prefix_cost = cost_system + cost_pinned
    summary_reserve = cfg.max_summary_tokens
    suffix_reserve = cfg.suffix_reserve_tokens
    priming_tokens = cfg.priming_tokens

    window_budget = (
        effective_ceiling - fixed_prefix_cost - summary_reserve - suffix_reserve - priming_tokens
    )
    if window_budget <= 0:
        raise ContextBudgetError(
            f"Fixed reserves exceed ceiling: effective={effective_ceiling}, "
            f"fixed_prefix={fixed_prefix_cost}, summary={summary_reserve}, "
            f"suffix={suffix_reserve}, priming={priming_tokens}"
        )

    # 5. Window Boundary (Algorithm A2: quantized onto chunk grid)
    def cost_fn(t: Turn) -> int:
        return memoized_turn_cost(t, tok)

    boundary = window_boundary(
        unpinned,
        window_budget,
        chunk_tokens=cfg.window_chunk_tokens,
        min_recent_turns=cfg.min_recent_turns,
        cost_fn=cost_fn,
        require_user_start=cfg.require_user_start,
    )
    window = list(unpinned[boundary:])
    dropped = list(unpinned[:boundary])

    # 6. Summarize Dropped Turns (Algorithm A6: pure function of dropped turns only)
    summary_text: str | None = None
    if dropped and cfg.max_summary_tokens > 0:
        redactor = RegexRedactor(mode=cfg.redaction_mode)
        summarizer = ExtractiveSummarizer(redactor=redactor)
        raw_summary = summarizer.summarize(dropped, max_tokens=cfg.max_summary_tokens, tok=tok)
        summary_text = raw_summary if raw_summary else None

    # 7. Dynamic Suffix: Query cost & Bounded Retrieval (Algorithm A9)
    query_tokens = tok.count(query)
    if query_tokens > suffix_reserve:
        if cfg.truncate_query:
            query = truncate_text_to_tokens(query, max_tokens=suffix_reserve, tokenizer=tok)
            query_tokens = tok.count(query)
        else:
            raise ContextBudgetError(
                f"User query ({query_tokens} tokens) exceeds suffix reserve ({suffix_reserve} tokens)"
            )

    retrieval_budget = max(0, suffix_reserve - query_tokens)
    retriever = BM25Retriever()
    hits = retriever.retrieve(
        query=query,
        candidates=dropped,
        k=cfg.retrieval_top_k,
        token_budget=retrieval_budget,
        tok=tok,
    )
    retrieval_tokens = sum(tok.count(h.get_effective_text()) for h in hits)

    # 8. Render Unified Messages (Algorithm A5)
    msgs = render(
        system_prompt=system_prompt,
        pinned=pinned,
        summary=summary_text or "",
        window=window,
        hits=hits,
        query=query,
        config=cfg,
    )

    validate_messages(msgs)

    # 9. Final Gate (Algorithm A5, Invariant I5)
    measured_tokens = gate(msgs, tok, effective_ceiling)
    if measured_tokens > effective_ceiling:
        # Recompute ONCE with safety margin deficit deducted
        deficit = (measured_tokens - effective_ceiling) + int(
            effective_ceiling * cfg.budget_safety_margin
        )
        new_window_budget = window_budget - deficit
        if new_window_budget <= 0:
            raise ContextBudgetError(
                f"Rendered prompt tokens ({measured_tokens}) exceed effective ceiling ({effective_ceiling})"
            )

        boundary = window_boundary(
            unpinned,
            new_window_budget,
            chunk_tokens=cfg.window_chunk_tokens,
            min_recent_turns=cfg.min_recent_turns,
            cost_fn=cost_fn,
            require_user_start=cfg.require_user_start,
        )
        window = list(unpinned[boundary:])
        dropped = list(unpinned[:boundary])

        if dropped and cfg.max_summary_tokens > 0:
            redactor = RegexRedactor(mode=cfg.redaction_mode)
            summarizer = ExtractiveSummarizer(redactor=redactor)
            raw_summary = summarizer.summarize(dropped, max_tokens=cfg.max_summary_tokens, tok=tok)
            summary_text = raw_summary if raw_summary else None
        else:
            summary_text = None

        hits = retriever.retrieve(
            query=query,
            candidates=dropped,
            k=cfg.retrieval_top_k,
            token_budget=retrieval_budget,
            tok=tok,
        )
        retrieval_tokens = sum(tok.count(h.get_effective_text()) for h in hits)

        msgs = render(
            system_prompt=system_prompt,
            pinned=pinned,
            summary=summary_text or "",
            window=window,
            hits=hits,
            query=query,
            config=cfg,
        )
        validate_messages(msgs)
        measured_tokens = gate(msgs, tok, effective_ceiling)
        if measured_tokens > effective_ceiling:
            raise ContextBudgetError(
                f"Rendered prompt tokens ({measured_tokens}) exceed effective ceiling ({effective_ceiling})"
            )

    # 10. Accounting & Assembly Report
    window_tokens = sum(cost_fn(t) for t in window)
    summary_tokens = tok.count(summary_text) if summary_text else 0
    estimated_total = (
        fixed_prefix_cost
        + summary_tokens
        + window_tokens
        + retrieval_tokens
        + query_tokens
        + priming_tokens
    )

    budget_plan = BudgetPlan(
        context_ceiling=cfg.context_ceiling,
        completion_reserve=cfg.completion_reserve,
        effective_budget=effective_ceiling,
        system_tokens=cost_system,
        pinned_tokens=cost_pinned,
        summary_tokens=summary_tokens,
        window_tokens=window_tokens,
        retrieval_tokens=retrieval_tokens,
        query_tokens=query_tokens,
        total_used_tokens=estimated_total,
        remaining_tokens=max(0, effective_ceiling - measured_tokens),
        measured_total_tokens=measured_tokens,
    )

    # Prefix stability calculation: prefix zone includes all messages except the final dynamic user message
    prefix_messages = msgs[:-1] if len(msgs) > 1 else msgs
    prefix_content = "".join(f"{m.role}:{m.content}\n" for m in prefix_messages)
    prefix_hash = hashlib.sha256(prefix_content.encode("utf-8")).hexdigest()
    summary_hash = (
        hashlib.sha256(summary_text.encode("utf-8")).hexdigest() if summary_text else None
    )
    cache_stable_tokens = tok.count_messages(prefix_messages, add_generation_prompt=False)

    report = AssemblyReport(
        tokenizer_id=tok.id,
        estimated_total_tokens=estimated_total,
        measured_total_tokens=measured_tokens,
        window_start_index=boundary,
        boundary_changed=False,
        compaction_occurred=boundary > 0,
        dropped_turn_ids=tuple(t.id for t in dropped),
        retrieved_turn_ids=tuple(h.id for h in hits),
        summary_hash=summary_hash,
        prefix_hash=prefix_hash,
        predicted_cache_stable_tokens=cache_stable_tokens,
    )

    ribbon_repr = (
        f"[sys: {cost_system}t] "
        f"[pin: {cost_pinned}t] "
        f"[sum: {summary_tokens}t] "
        f"[win({len(window)}): {window_tokens}t] "
        f"| [ret({len(hits)}): {retrieval_tokens}t] "
        f"[q: {query_tokens}t] "
        f"=> {measured_tokens}/{effective_ceiling}t"
    )

    return AssembledContext(
        messages=msgs,
        budget_plan=budget_plan,
        ribbon_representation=ribbon_repr,
        window_turns=window,
        retrieved_turns=tuple(hits),
        retrieved_turn=hits[0] if hits else None,
        dropped_turns=dropped,
        summary=summary_text,
        pinned_turns=pinned,
        report=report,
    )
