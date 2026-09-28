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

### Changed
- Refactored `pyproject.toml` to keep core dependency-light (`rank-bm25`, `httpx`) with optional extras (`[hf]`, `[tiktoken]`, `[rich]`, `[dense]`, `[otel]`, `[postgres]`, `[redis]`, `[bench]`, `[dev]`).
- Synchronized `requirements.txt` to point to `-e .[dev]`.

### Removed
- Legacy orphan directory `context_engine/`.

## [0.1.0] - 2026-09-27
- Initial baseline release of Context-Engineer: deterministic prefix-cache ribbon context manager with 5 layers (Cap, Pin, Retrieve, Window, Summarize).
