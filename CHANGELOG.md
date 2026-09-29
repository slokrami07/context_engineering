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
- Phase 3: Protocol interface definitions (`DenseEmbedder`, `Store`, `Backend`, `ChatResult`, `CacheStats` in `context_engineer.protocols`).
- Phase 3: Pluggable retrieval subsystem (`context_engineer.retrievers`):
  - `BM25Retriever`: Hardened BM25 with SHA-256 candidate index caching (`R4`), small-corpus IDF flooring (`R5`), query expansion (`R6`), tie-breaking by score then ID (`R8`), identifier splitting, and hard token bounding.
  - `DenseRetriever` & `OfflineSemanticEmbedder`: Offline semantic vector projection using local hashing/numpy with zero external downloads.
  - `HybridRetriever`: Reciprocal Rank Fusion (RRF: $\sum \frac{w}{60 + \text{rank}}$) fusing lexical BM25 and dense semantic ranking with dense/lexical tie-breaking.
- Phase 3: Thread-safe `InMemoryStore` and updated `SqliteStore` satisfying the `Store` protocol (`context_engineer.store`).
- Phase 3: `MockBackend` conforming to the `Backend` protocol with call recording and configurable cache stats simulation (`context_engineer.backends`).
- Phase 3: Formal contract test suites (`tests/contracts/`) covering Tokenizer (4 tests), Retriever across BM25/Dense/Hybrid (18 tests), Summarizer (3 tests), Redactor (3 tests), Store across InMemory/Sqlite (10 tests), and Backend (3 tests).
- Phase 3: Acceptance tests for hybrid retrieval (`tests/test_hybrid_retrieval.py`) proving zero-change pipeline protocol swapping, hard token budget bounding, and Hybrid beating BM25 on paraphrase scenarios with no token overlap.
- Phase 4: Measurement and credibility benchmark suite (`benchmarks/`):
  - Primary progression protocol measuring per-request reuse distributions (median, p10, p90) as history expands by 1-2 turns per request (Algorithm A13, LG2).
  - Fair baselines built from shared components: `Plain Left-Truncation`, honest `Chronological RAG` using `BM25Retriever`, `Suffix-RAG (Unquantized Window)`, `Context-Engineer (Full)`, and `Raw Append-Only` tracking ceiling overflow (LG1, LG3).
  - Procedural scenario generators with strictly alternating roles, deterministic seeding (`random.Random`), near-duplicate distractors (`shard-19` vs `shard-91`), and multi-needle probes across systems/support/code domains (A13, PR1-PR3, EV2-EV3).
  - Cache probe module (`benchmarks/cache_probe.py`) measuring or simulating token prefix reuse across consecutive turns (Algorithm A12).
  - Evaluation gates: `fact_present`, `fact_recalled` (marked `SKIPPED` in offline/mock mode to eliminate fabricated model responses, EV1), `prefix_stability` (I1/I2), and `budget_enforced` (I5).
  - Artifact serialization generating JSON and Markdown reports (`results/<date>-<backend>.json` and `.md`) with an explicit "Where Context-Engineer Loses" tradeoff analysis (X4, X6).
  - Dedicated benchmark test suite `tests/test_benchmarks.py` (5 tests) covering scenario generation, cache probe, offline gate skipping, and baseline execution.

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
- Phase 3: `assemble_context` accepts pluggable `retriever: Retriever | None` and `summarizer: Summarizer | None`, allowing zero-pipeline-change implementation swapping.
- Phase 4: `main.py` updated to run `benchmarks.run` instead of legacy `harness`.

### Removed
- Phase 0: Legacy orphan directory `context_engine/`.
- Phase 4: Legacy `harness/` package, eliminating fabricated offline evaluations (EV1) and keyword-sniffing strawman baseline (LG1).

## [0.1.0] - 2026-09-27
- Initial baseline release of Context-Engineer: deterministic prefix-cache ribbon context manager with 5 layers (Cap, Pin, Retrieve, Window, Summarize).
