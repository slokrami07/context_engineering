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

### Changed
- Phase 0: Refactored `pyproject.toml` to keep core dependency-light (`rank-bm25`, `httpx`) with optional extras (`[hf]`, `[tiktoken]`, `[rich]`, `[dense]`, `[otel]`, `[postgres]`, `[redis]`, `[bench]`, `[dev]`).
- Phase 0: Synchronized `requirements.txt` to point to `-e .[dev]`.
- Phase 1: `ContextConfig` is now frozen and hashable, with strict parameter validation in `__post_init__` (`G2`).
- Phase 1: Deprecated `Turn.tool_output` with `DeprecationWarning` in favor of structured tool calls and `kind="tool_result"` (`T1`).
- Phase 1: Layers `cap` and `retrieve` refactored to use `dataclasses.replace` rather than mutating `Turn` in-place.
- Phase 1: Tokenizer resolution enforces offline security by default (`local_files_only=True`, `trust_remote_code=False`) (`K1`, `I10`).

### Removed
- Legacy orphan directory `context_engine/`.

## [0.1.0] - 2026-09-27
- Initial baseline release of Context-Engineer: deterministic prefix-cache ribbon context manager with 5 layers (Cap, Pin, Retrieve, Window, Summarize).
