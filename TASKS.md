# Tasks

**Owner**: lead-agent (exclusively). Other agents read this file but do not write to it.

Each agent has a separate plan file with detailed checklists:
- [TASKS-core.md](TASKS-core.md) -- core-agent tasks
- [TASKS-feature.md](TASKS-feature.md) -- feature-agent tasks
- [TASKS-qa.md](TASKS-qa.md) -- qa-agent tasks

## Format

Each task has: ID, status, agent, title, details, and optional dependencies.

**Statuses**: `open`, `in-progress`, `done`, `blocked`, `deferred`

## Open Tasks

| ID | Agent | Title | Details | Depends on |
|----|-------|-------|---------|------------|
| T1 | qa-agent | Set up test infrastructure | Create conftest.py with shared fixtures, configure pytest | - |
| T2 | core-agent | Implement core algorithms | Build core computation modules with type annotations | - |
| T3 | feature-agent | Implement business logic | Build decision/workflow logic on top of core modules | T2 |
| T4 | qa-agent | Write unit tests | Cover all public methods with unit tests | T2, T3 |
| T5 | lead-agent | Set up logging infrastructure | Structured logging, log directory, diagnostic mode | - |
| T6 | qa-agent | Set up benchmarking | Reproducible benchmark runner with seed management | T2, T3 |
| T7 | lead-agent | Build log analysis tool | CLI tool for log inspection and problem detection | T5 |

## In Progress

| ID | Agent | Title | Status | Notes |
|----|-------|-------|--------|-------|
| - | - | - | - | - |

## Done

| ID | Agent | Title | Result |
|----|-------|-------|--------|
| - | - | - | - |
