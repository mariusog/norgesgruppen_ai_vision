# Project Best Practices

## Project Tooling

Configure these for your language/stack. All commands throughout this file and in agent configs reference these.

| Tool | Command | Notes |
|------|---------|-------|
| **Test (fast)** | `python -m pytest tests/ -q --tb=line -m "not slow" 2>&1 \| tail -20` | Pipe through tail. Use quiet mode. |
| **Test (debug)** | `python -m pytest tests/path/file.py::test_name -q --tb=short 2>&1 \| tail -40` | Only for investigating a specific failure. |
| **Lint** | `ruff check <files>` | Auto-fix: `ruff check --fix <files>` |
| **Format** | `ruff format <files>` | Check only: `ruff format --check <files>` |
| **Type check** | `mypy <files>` | |
| **Security scan** | `bandit -r src/ -ll` | Dependencies: `pip-audit` |
| **Log analysis** | `<your-tool> <log> --brief 2>&1 \| tail -15` | Configure per-project. See Log-Reading Workflow below. |
| **Benchmark** | `<your-tool> --diagnostics` | Configure per-project. Results go to `docs/`. |
| **Constants file** | `src/constants.py` | All magic numbers and tuning parameters. |
| **Test fixtures** | `tests/conftest.py` | Shared factories and setup/teardown. |
| **Source extension** | `.py` | |

**To adapt for another language**: Replace the commands above. The rest of this file (principles, workflows, coordination) is language-neutral.

<details><summary>Common alternative stacks</summary>

| Tool | TypeScript/Node | Go | Rust | Ruby/Rails |
|------|----------------|-----|------|------------|
| Test (fast) | `npx jest --silent 2>&1 \| tail -20` | `go test ./... 2>&1 \| tail -20` | `cargo test 2>&1 \| tail -20` | `bundle exec rspec --format progress 2>&1 \| tail -20` |
| Lint | `npx eslint <files>` | `golangci-lint run` | `cargo clippy` | `bundle exec rubocop <files>` |
| Format | `npx prettier --write <files>` | `gofmt -w <files>` | `cargo fmt` | `bundle exec rubocop -a <files>` |
| Type check | (built-in with tsc) | (built-in) | (built-in) | `bundle exec srb tc` (Sorbet) |
| Security | `npm audit` | `govulncheck ./...` | `cargo audit` | `bundle audit check --update` |
| Constants file | `src/constants.ts` | `internal/constants.go` | `src/constants.rs` | `config/constants.rb` |
| Test fixtures | `tests/helpers.ts` | `testutil/helpers.go` | `tests/common/mod.rs` | `spec/factories/` (FactoryBot) |
| Source extension | `.ts` | `.go` | `.rs` | `.rb` |

</details>

## AI Agent Ground Rules

Read this section FIRST. These are the rules that save you from wasting tokens and making common mistakes.

<important>

### NEVER Do These

- **NEVER use verbose test output** -- use quiet mode and pipe through `tail`
- **NEVER read raw CSV log files** -- read the summary report first (see Benchmarking section), use the analysis tool for drill-down
- **NEVER parse stdout to extract results** -- read the generated report files instead
- **NEVER use JSON lines for high-volume data** -- use CSV (4x more token-efficient)
- **NEVER use bare print/console.log for operational output** -- use the language's logging framework
- **NEVER dump full file contents when a summary exists** -- read summaries first, drill down on demand
- **NEVER use unseeded randomness** -- all randomized code MUST accept a `seed` parameter
- **NEVER modify files owned by another agent** -- add a `BLOCKED` tag in your plan file instead

### ALWAYS Do These

- **ALWAYS pipe command output through `tail`** -- bound your token consumption
- **ALWAYS read TASKS.md and your plan file before starting work** -- check for assigned tasks and escalations
- **ALWAYS run tests before committing** -- zero tolerance for test failures
- **ALWAYS include before/after metrics** when reporting optimization results
- **ALWAYS log the seed** in every run so results can be replicated

</important>

### Token Budget Awareness

You are an AI agent with a finite context window. Optimize for it:

| Action | Token-Efficient Way | Token-Wasteful Way |
|--------|--------------------|--------------------|
| Check test results | Fast test command from Tooling table | Verbose test output (unbounded) |
| Read benchmark results | Read the summary report in `docs/` | Parse stdout from benchmark run |
| Inspect a log | Analysis tool with `--brief` | Read the raw CSV file |
| Compare two runs | Analysis tool with `--compare` | Read both files and diff manually |
| Check for problems | Analysis tool with `--problems` | Read full summary + all data |

### Codebase Orientation (when starting in a new project)

When dropped into an unfamiliar codebase, read in this order:

1. **CLAUDE.md** -- this file (project rules, structure, conventions)
2. **TASKS.md** and your **per-agent plan file** (TASKS-core.md, TASKS-feature.md, or TASKS-qa.md) -- task assignments and detailed checklists
3. **Project structure** -- `ls` the source and test directories
4. **Constants file** -- all tuning parameters and configuration values
5. **Test fixtures** -- understand the shared setup and factory functions
6. **The specific files related to your task** -- only then dive into source code

Do NOT read every file. Read the minimum needed for your task.

### Skill Selection Guide

**Before writing code**, use this decision tree:

```
Planning a new feature or major change?
+-- Need to define requirements? -> write-a-prd
+-- Have a PRD, need an implementation plan? -> prd-to-plan
+-- Have a PRD, need GitHub issues? -> prd-to-issues
+-- Want to stress-test a design? -> grill-me
+-- Need architecture guidance? -> project-architecture
+-- Want to find refactoring opportunities? -> improve-codebase-architecture
+-- Designing a data processing workflow? -> data-pipeline
+-- Designing error handling strategy? -> error-handling
+-- Building a new feature test-first? -> tdd-cycle
```

**After writing or modifying code**, use this decision tree:

```
Did you write new code?
+-- Yes: Run test-coverage (verify tests exist for new public methods)
+-- Need integration or API tests? -> integration-testing
+-- Need browser or e2e tests? -> browser-testing
+-- Did tests fail?
|   +-- Yes: Run debugging skill. Do NOT proceed until green.
|   +-- Need text-based debug tools? -> debug-visualization
+-- Is this a performance-sensitive change?
|   +-- Yes: Run performance-optimization
|   +-- Need caching? -> caching-strategies
+-- Need to add or improve logging? -> logging-observability
+-- Want to clean up code structure? -> refactor
+-- Ready to ship?
|   +-- Yes: Run production-quality (orchestrates lint, refactor, test-coverage, security-scan, code-review, update-documentation)
+-- Quick quality check only?
    +-- Yes: Run lint + code-review
```

**Shipping and maintenance:**

```
+-- Creating a pull request? -> pr-workflow
+-- Adding or updating dependencies? -> dependency-management
+-- Need to update docs? -> update-documentation
+-- Writing or updating README? -> readme-standards
+-- Dealing with randomness or seeds? -> reproducibility
+-- Need a security audit? -> security-scan
```

For new features, use `tdd-cycle` (write tests first, then implement).

### Skill Design Conventions

- **Description field**: Lead with the trigger condition ("Use when..."), not a summary of what the skill does. The description is how Claude decides whether to activate the skill.
- **Gotchas section**: Every skill should accumulate a `## Gotchas` section documenting real failure patterns observed in practice. This is the highest-signal content in a skill — add entries when Claude fails at something the skill should handle. The `production-quality` self-improvement step (Step 11) is the right time to capture new Gotchas.

## Project Structure

Adapt to your language. The key principle: **source and test directories mirror each other**.

```
your_project/
+-- src/                        # Main package
|   +-- constants.*             # Named constants (tuning parameters)
|   +-- ...                     # Your modules
+-- tests/
|   +-- fixtures/helpers        # Shared test setup and factories
|   +-- ...                     # Mirror source structure exactly
+-- logs/                       # Runtime logs and debug data
+-- docs/                       # Generated reports
+-- CLAUDE.md                   # This file
+-- TASKS.md                    # Task tracking board
```

## Code Quality Standards

### Hard Limits

- **300 lines max per file** (source and test)
- **200 lines max per class/struct/module**
- **30 lines max per method/function** (excluding doc comments)
- **No magic numbers** -- all thresholds in the constants file
- **Type annotations required** on all public function signatures (where the language supports it)

### SOLID Principles

- **SRP**: Every class/function has one responsibility. If you can say "and", split it.
- **OCP**: Behavior is extensible without modifying existing code.
- **LSP**: Subtypes are drop-in replacements for their base types.
- **ISP**: Depend on small, focused interfaces -- not large monolithic ones. No wildcard imports.
- **DIP**: High-level modules depend on abstractions, not low-level implementation details.

### Law of Demeter

- Max one level of chaining: `self.state.getDistance()` OK, `self.state.cache[pos].get(target)` NOT OK.

### Safety

- All search/exploration functions have bounded iteration (`maxSteps`, `maxCells`, etc.)
- No unbounded loops or recursion without explicit depth limits
- Validate at system boundaries: positions within bounds, inputs within limits

## Testing Standards

### Coverage Requirements

**Every public method and function MUST have at least one dedicated test.** Non-negotiable.

Test naming: `test_<method_name>_<scenario>` -- e.g., `test_calculate_score_empty_input_returns_zero`

### Unit Test Checklist (per method)

1. **Happy path** -- normal input, expected output
2. **Edge cases** -- empty inputs, zero values, boundary conditions
3. **Error conditions** -- invalid input, unreachable targets, overflow
4. **State mutations** -- verify side effects

### Test Quality Rules

- **Arrange-Act-Assert** pattern in every test
- **One assertion per concept** -- test one behavior, not five
- **No test interdependence** -- each test creates its own state via fixtures/factories
- **Deterministic** -- no random inputs without fixed seeds
- **Descriptive names** -- `test_search_stops_at_max_depth` not `test_search_1`

### Reproducibility

- All randomized processes MUST accept a `seed` parameter
- Benchmark runs MUST be reproducible with the same seed
- Log the seed in every run so results can be replicated
- Test data must be deterministic -- use factory functions, not random generators

### Running Tests

Use the **Test (fast)** command from the Tooling table. When debugging a failure, use **Test (debug)**.

**IMPORTANT**: Always use quiet mode and pipe through `tail`. Never use verbose output.

### When Tests Fail

1. Read the failure output (the `tail` you already have)
2. If the error is clear, fix it and rerun
3. If unclear, rerun the single failing test with more detail (see Test debug command)
4. If the failure is in another agent's code, do NOT fix it -- add a task to TASKS.md
5. If a test is flaky (passes sometimes, fails sometimes), it has a non-determinism bug -- fix the seed

## Logging and Debug Data

### Token-Efficient Log Formats

| Data Type | Format | Why |
|-----------|--------|-----|
| Per-step tabular data | **CSV** (short column names) | Headers once, 4x cheaper than JSON lines |
| Run metadata | **JSON** | One-off, structured, small |
| Pre-computed summaries | **Markdown** | Agents read directly as files |

- **NEVER** use JSON lines for high-volume per-step data (keys repeat every row)
- Use short CSV column names: `ts,id,op,val` not `timestamp,entity_id,operation,value`
- Log files go in `logs/` with timestamps: `<name>_<YYYY-MM-DD_HH-MM-SS>.{csv,json}`

### Three-Tier Log Architecture

1. **Tier 1 -- Summary report** (agents read this FIRST): Pre-computed markdown in `docs/` (e.g., `benchmark_results.md`). Contains scores, key metrics, auto-detected problems. Under 40 lines.
2. **Tier 2 -- CSV detail log**: Per-step data in `logs/`. Agents drill into this ONLY when Tier 1 flags a problem. Never read raw CSV directly -- use the analysis tool.
3. **Tier 3 -- JSON metadata**: Config, seed, environment. One file per run.

### Agent Log-Reading Workflow

```sh
# Step 1: Read the summary FIRST (5-15 lines)
cat docs/<summary_report>.md

# Step 2: If problems, get the anomaly list (use Log analysis command from Tooling table)
<log_analysis_command> <log> --problems 2>&1 | tail -20

# Step 3: Drill into a specific record or category
<log_analysis_command> <log> --filter <id> --brief 2>&1 | tail -15

# Step 4: Only if still unclear, look at raw step range
<log_analysis_command> <log> --steps 40-50 2>&1 | tail -20
```

Agents should almost NEVER need to go past Step 2.

### Visualization

- **Text-only**: ASCII grids, markdown tables, compact summaries
- Agents can't open browsers or view images -- all debug tools must work in terminal
- Visualization reads from log files -- never couple to runtime
- Run-length encode timelines: `retry x3 | success | idle x12 | timeout x2`

## Benchmarking

- Write results to a summary report in `docs/` -- agents read the file, NEVER parse stdout
- Always run with `--diagnostics` after optimization changes
- Include before/after scores and problem counts in task notes
- Use multiple seeds for statistical comparison (10+ seeds recommended)
- Comparisons MUST use the same seed list to be valid
- Report only changed metrics in diffs (skip unchanged values to save tokens)

## Multi-Agent Coordination Protocol

When multiple agents run in parallel (via worktrees), they MUST follow this protocol.

### Execution Model

1. **Lead-agent runs first** -- diagnoses, plans, creates tasks
2. **Core, feature, and QA agents start in parallel** -- each works on assigned tasks independently
3. **Lead-agent monitors and validates** -- reads agent plan files, handles escalations, merges branches

### Task Architecture

| File | Owner | Purpose |
|------|-------|---------|
| `TASKS.md` | lead-agent (exclusively) | Master task board with assignments, statuses, and result summaries |
| `TASKS-core.md` | core-agent | Detailed checklists, progress, and results for core tasks |
| `TASKS-feature.md` | feature-agent | Detailed checklists, progress, and results for feature tasks |
| `TASKS-qa.md` | qa-agent | Detailed checklists, audit reports, and escalations |

**Each agent only writes to their own plan file.** This eliminates merge conflicts on shared files.

### Task Workflow

1. Lead creates tasks in `TASKS.md` and writes detailed checklists in the appropriate agent plan file
2. Agents read `TASKS.md` for assignments, then work from their own plan file
3. Agents check off items (using `- [x]` checkboxes) and write results in their plan file as they progress
4. Lead validates by reading agent plan files, running `code-review`, and verifying metrics
5. Lead marks the task as `done` in `TASKS.md` after validation — a task is not officially complete until this step

### Task Result Format

Every completed task MUST report in the agent's plan file:
- **What changed**: One sentence describing the code change
- **Metrics**: Before/after numbers for the targeted metric
- **Tests**: Pass count, any new tests added

### Escalation

- If blocked by another agent's code, mark the task as **BLOCKED** in your plan file with a description
- If QA finds a critical issue (data loss, security hole, crash), mark as **CRITICAL** in `TASKS-qa.md`
- Lead-agent monitors plan files, triages blockers, and updates plan files when resolved
- Do NOT silently stall -- always surface the blocker

Escalation entry format in your plan file:

```
### T5: Implement retry logic
**Status**: BLOCKED
**Blocker**: core-agent — `src/data/connection.py` missing `retry()` method needed by handler
**Since**: 2025-03-19
```

Include: which agent/file is blocking, what's missing, and when you escalated.

### Conflict Prevention

- Each agent only modifies their owned files -- never touch another agent's code
- If you discover work needed in another agent's files, add it to your plan file with a `BLOCKED` tag
- One task at a time -- finish or abandon before claiming another
- Lead-agent handles all cross-cutting changes

## File Ownership

| Agent | Owned Files | Role |
|-------|-------------|------|
| lead-agent | Entry points, constants, `TASKS.md`, `TASKS-*.md` (creates), `CLAUDE.md`, `.claude/agents/` | Architecture, coordination, cross-cutting changes |
| core-agent | Core algorithm and data modules, `TASKS-core.md` (fills out) | Algorithms, data structures, computation |
| feature-agent | Feature/business logic modules, `TASKS-feature.md` (fills out) | Decision logic, workflows, rules |
| qa-agent | `tests/`, benchmarks, `docs/`, `TASKS-qa.md` (fills out) | Testing, benchmarking, quality enforcement |

The lead-agent has cross-cutting authority -- it may modify any file when a fix spans multiple agents' boundaries.

## Git Conventions

### Commit Messages

- Short, single-sentence, present tense: `Fix cache invalidation on round reset`
- Focus on WHY, not WHAT: `Prevent stale cache after config reload` not `Change line 42 in cache.py`
- One logical change per commit

### Staging

- **Stage specific files by name** -- never use `git add .` or `git add -A` (risks committing secrets, logs, cache files)
- Check `git status` before committing -- verify only intended files are staged
- Never commit files matching `.gitignore` patterns

### Branches

- `main` is the stable branch -- all tests pass, benchmarks meet baselines
- Feature branches: `<agent>/<task-id>-<short-description>` e.g. `core/T12-cache-invalidation`
- Never force-push to `main`

## Bootstrapping This Template

### Quick Start (automated)

```bash
./bootstrap.sh /path/to/new-project --lang python
```

This copies the full structure, creates directories, and sets up language-specific config. Supported languages: `python`, `typescript`, `go`, `rust`, `ruby`.

### Manual Setup

1. Copy `.claude/` into your project root
2. Copy this CLAUDE.md -- update the **Project Tooling** table for your language/stack
3. Replace the **Project Structure** section with your actual layout
4. Copy `templates/.gitignore` (adapt language-specific patterns)
5. Copy `templates/TASKS.md` for task tracking and `templates/TASKS-agent.md` to create per-agent plan files (`TASKS-core.md`, `TASKS-feature.md`, `TASKS-qa.md`)
6. If Python: copy `templates/pyproject.toml`, `templates/conftest.py`, `templates/constants.py`
7. If other language: create equivalent config, test fixtures, and constants files
8. If using Claude Code hooks: copy `templates/.claude/settings.json` and `templates/.claude/hooks/`
9. Copy `templates/.github/workflows/ci.yml` for CI pipeline (adapt commands to match your Tooling table)
10. Update the **File Ownership** table with your actual file paths
