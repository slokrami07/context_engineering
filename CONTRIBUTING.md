# Contributing to Context-Engineer

Thank you for your interest in contributing to Context-Engineer!

## Development Principles & Invariants

Context-Engineer is designed with a strict set of core invariants:
- **Append-only prefix stability (I1):** Do not break KV-cache reuse. The prefix zone must be stable across turns.
- **Query independence (I2, I3):** Prefix zone and window boundary must never depend on the query or retrieval size.
- **Budget safety (I5):** Never return an over-budget prompt. Always raise `ContextBudgetError`.
- **Structural validity (I6):** Never split tool call / tool result pairs.
- **Determinism (I7):** Identical inputs must yield identical bytes on any machine.
- **Pure, idempotent capping (I8):** `cap(cap(t)) == cap(t)`.

## Development Setup

1. Clone repository and install development dependencies:
   ```bash
   git clone https://github.com/slokrami07/context_engineering.git
   cd context_engineering
   uv venv
   uv pip install -e ".[dev]"
   ```

2. Run test suite:
   ```bash
   pytest
   ```

3. Code formatting and linting:
   ```bash
   ruff check .
   ruff format --check .
   ```

4. Type checking:
   ```bash
   mypy src/
   ```

## Pull Request Guidelines

- Follow conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
- Every bugfix must start with a failing test reproducing the issue.
- Maintain characterization and property-based test coverage.
- Record any intentional behavior changes in `CHANGELOG.md`.
