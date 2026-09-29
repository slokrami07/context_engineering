# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0: Baseline hygiene, layout consolidation, standard open source governance (LICENSE Apache-2.0, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY).
- Phase 0: PEP 561 `py.typed` marker.
- Phase 0: Dedicated CLI module `context_engineer.cli.chat`.
- Phase 0: Characterization tests pinning current layer behaviors.
- Phase 0: Multi-OS GitHub Actions CI workflow (linting, type checking, test matrix on Python 3.10-3.13 on ubuntu and macos).
- Phase 1: Domain exception hierarchy (`ConfigError`, `HistoryError`, `ContextBudgetError`, `StructuralError`, `TokenizerError`, `TokenizerUnavailable`).
- Phase 1: Immutable data structures (`Turn`, `ToolCall`, `Message`, `BudgetPlan`, `AssemblyReport`) with `frozen=True` and `slots=True`.
- Phase 1: History sequence validation (`validate_history`) enforcing monotonic non-duplicate turn IDs.
- Phase 1: Hardened `tokenizers/` package with `ApproxTokenizer`, `TiktokenTokenizer`, `HFTokenizer`, and thread-safe `resolve_tokenizer`.
- Phase 1: Boundary-safe `slice_head_tail` with negative compression prevention (`K5`), `tail >= 1` requirement (`K4`), and multi-byte UTF-8 replacement character prevention (`K6`).
- Phase 1: Comprehensive test suites for types, config validation, and tokenizer offline foundations (`tests/test_config.py`, `tests/test_types.py`, `tests/test_tokenizer_foundations.py`).
- Phase 2: Protocols interface definition (`Tokenizer`, `Retriever`, `Summarizer`, `Redactor` in `context_engineer.protocols`).
- Phase 2: Entity scrubbing subsystem (`RegexRedactor`, `patterns.py`) with negative lookbehind path guards and digit-bound resource guards (Algorithm A7, S4-S6).
- Phase 2: Domain-agnostic `ExtractiveSummarizer` operating purely on dropped turns without domain boilerplate or turn counters (Algorithm A6, S1-S3, S8).
- Phase 2: Pure, idempotent Cap Layer v2 (`cap.py`) supporting JSON preservation, line elision, and token slicing with raw content preservation (Algorithm A10, C1-C6, I8).
- Phase 2: Stateless quantized window boundary and structural legal boundary checking (`window.py`) preventing orphaned tool results and stabilizing cache prefix (Algorithms A2 & A4, W1-W7, I1, I4, I6).
- Phase 2: Unified rendering and final template gate (`render.py`) eliminating mid-dialogue system messages and role alternation hazards (Algorithm A5, PL4, PL5, S7, P4, I5, I6).
- Phase 2: Pipeline rewrite (`pipeline.py`) computing window budgets from fixed reserves independent of query or retrieval size (Algorithms A1 & A3, PL1-PL7, I2, I3, I9).
- Phase 2: Property-based test suite (`tests/test_invariants.py`) covering Invariants I1 through I10.
- Phase 2: Simulated prefix reuse test suite (`tests/test_prefix_reuse.py`) reporting 100.00% reuse between consecutive non-compaction turns.
- Phase 2: Acceptance test (`tests/test_domain_strings.py`) asserting zero domain-specific strings in library source code.

### Changed
- Phase 0: Refactored `pyproject.toml` to keep core dependency-light (`rank-bm25`, `httpx`) with optional extras (`[hf]`, `[tiktoken]`, `[rich]`, `[dense]`, `[otel]`, `[postgres]`, `[redis]`, `[bench]`, `[dev]`).
- Phase 0: Synchronized `requirements.txt` to point to `-e .[dev]`.
- Phase 1: `ContextConfig` is now frozen and hashable, with strict parameter validation in `__post_init__` (`G2`).
- Phase 1: Deprecated `Turn.tool_output` with `DeprecationWarning` in favor of structured tool calls and `kind="tool_result"` (`T1`).
- Phase 1: Layers `cap` and `retrieve` refactored to use `dataclasses.replace` rather than mutating `Turn` in-place.
- Phase 1: Tokenizer resolution enforces offline security by default (`local_files_only=True`, `trust_remote_code=False`) (`K1`, `I10`).
- Phase 2: Context ribbon rendering restructured to fuse system prompt, pinned facts, and narrative summary into a single head `system` message, avoiding middle-dialogue system messages (`PL5`, `S7`).
- Phase 2: Memory retrieval is now read-only over dropped turns, hard-bounded to `suffix_reserve - query_tokens`, and does not mutate the window boundary (`R1`-`R3`, `PL1`-`PL3`).
- Phase 2: Capping now operates idempotently before windowing, preserving invariant pinned turns without modification (`C1`, `P3`, `I8`).

### Removed
- Legacy orphan directory `context_engine/`.

## [0.1.0] - 2026-09-27
- Initial baseline release of Context-Engineer: deterministic prefix-cache ribbon context manager with 5 layers (Cap, Pin, Retrieve, Window, Summarize).
