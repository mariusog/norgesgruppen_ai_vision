---
name: production-quality
description: Use when the user asks to clean up code, improve code quality, prepare code for production, or run the production quality routine. Orchestrates lint, refactor, test-coverage, security-scan, code-review, and update-documentation skills to bring code to production quality.
---

# Production Quality Code Routine

Orchestrates multiple quality skills to bring all changed files up to production quality level.

## Scope

This routine applies to all files changed in the current branch compared to `main` (or `origin/main`).
Identify changed files with: `git diff --name-only origin/main...HEAD`

### File Type Considerations

Categorize changed files to determine which steps apply:

| File Type | Lint | Refactor | Test Coverage | Security | Code Review |
|-----------|------|----------|---------------|----------|-------------|
| Source code | Yes | Yes | Unit + integration tests | Static analysis | Yes |
| Test helpers/fixtures | Yes | Minimal | Verified by running tests | N/A | Yes |
| Config (`.yml`, `.json`, `.toml`) | N/A | N/A | Integration tests | Review secrets | Verify correctness |
| CI/Docker | N/A | N/A | Run CI pipeline | Review exposure | Verify correctness |
| Markdown (`.md`) | N/A | N/A | N/A | N/A | Verify accuracy |

For configuration-only changes, focus on: baseline tests passing, security review, and documentation updates.

## Step 1: Baseline Test Run

Run the `test-coverage` skill to ensure we start green:
- Identifies all relevant test files for changed code
- Runs baseline tests
- **If tests fail, STOP** -- do not proceed with quality improvements on broken code

## Step 2: Lint Cleanup

Run the `lint` skill:
- Linter for code issues
- Formatter for consistent style
- Type checker for annotation errors
- Do NOT disable any rules

## Step 3: Refactor for Clarity

Run the `refactor` skill:
- Replace conditionals with lookup tables where appropriate
- Use early returns and guard clauses
- Extract magic numbers and strings to constants
- Remove unnecessary comments

## Step 4: Test Coverage Check

Run the `test-coverage` skill again to:
- Verify all new public functions have tests
- Check edge cases and error paths
- Add missing tests if coverage is insufficient
- Verify reproducibility (same inputs -> same outputs)

## Step 5: Security Scan

Run the `security-scan` skill:
- Static analysis for code security issues
- Dependency audit for known vulnerabilities

## Step 6: Logging and Observability Check

Verify that changed code follows logging standards:
- Uses the language's logging framework, not bare print/console output
- Key operations produce structured log output
- Diagnostic mode available for verbose recording
- Sufficient data logged for post-mortem debugging

How to check:
1. Search for bare print/console output in changed source files: `grep -rn 'print(' src/` (adapt pattern for your language)
2. Verify logging imports exist in modules that perform I/O or key operations
3. Check that error handling blocks include log statements (not just silent catches)
4. If the project has a diagnostic/debug mode, verify it can be toggled via config

If changed files are config-only or documentation, skip this step.

## Step 7: Commit Changes

Commit the work in small, structured commits:
- One logical change per commit
- Stage specific files (never `git add .`)
- Use short, single-sentence commit messages in present tense

## Step 8: Code Review Loop

Run the `code-review` skill iteratively:
- Review for SOLID violations, language best practices, performance issues
- Fix the most critical issues found
- Commit each fix separately with clear messages
- Run tests after each fix
- Repeat until no critical issues remain

## Step 9: Update Documentation

Run the `update-documentation` skill:
- Update README if user-facing behavior changed
- Update architecture docs if structural changes were made
- Update doc comments for changed public APIs

## Step 10: Final Checks

Run all quality checks one final time using the commands from the CLAUDE.md Tooling table:
- Linter: verify no style issues
- Formatter: verify consistent formatting
- Tests: verify all pass
- Security: verify no issues

Do NOT disable any rules or remove any test functions.

## Step 11: Skill Self-Improvement

Review this run and improve the skills themselves:

### Identify Gaps
- Were any steps unclear or incomplete?
- Did you discover checks that should be added?
- Were there quality issues the skills didn't catch?
- Did documentation updates get missed that should be automated?

### Identify Obsolete Steps
- Are any steps now handled automatically by tooling or linters?
- Have new tools or packages made certain manual checks redundant?
- Are there steps that consistently yield no findings and can be removed?
- Can multiple steps be consolidated?

### Capture Gotchas
- Did any skill fail to catch an issue it should have? Add it to that skill's `## Gotchas` section
- Did Claude make a predictable mistake (e.g., testing implementation details, over-refactoring)? Document the pattern
- Gotchas are the highest-signal content in a skill — they prevent repeated failures

### Capture New Patterns
- New refactoring patterns worth adding as examples
- New anti-patterns to warn against
- Project-specific conventions to document

### Update Skills
If improvements are identified:
1. Update the relevant skill file(s) in `.claude/skills/`
2. Keep changes focused and actionable
3. Add concrete examples where helpful

If the run was smooth and no gaps were found, skip this step.

## Gotchas

- **Skipping steps for "simple" changes**: Even small changes can introduce regressions. Run the full routine — the steps that feel unnecessary are often the ones that catch surprises.
- **Markdown-only changes still need review**: Configuration, documentation, and skill files can contain stale references, broken links, or inconsistent guidance. The code review step applies to all file types.
- **Scoring N/A skills as 100**: When a sub-skill is not applicable (e.g., no source code to lint), redistribute its weight rather than giving it a perfect score — otherwise the aggregate is inflated.

## Scoring (0-100)

The production readiness score is the weighted average of sub-skill scores:

| Sub-skill | Weight | Source |
|-----------|--------|--------|
| Test coverage | 25% | Step 4 score |
| Code review | 25% | Step 8 score |
| Security scan | 20% | Step 5 score |
| Lint | 15% | Step 2 score |
| Refactor | 15% | Step 3 score |

**Production readiness = (test × 0.25) + (review × 0.25) + (security × 0.20) + (lint × 0.15) + (refactor × 0.15)**

If a sub-skill is not applicable (e.g., no source code to lint), redistribute its weight equally among the remaining skills.

| Score | Interpretation |
|-------|---------------|
| 90-100 | Production-ready -- ship with confidence |
| 70-89 | Nearly ready -- address remaining major issues |
| 50-69 | Not ready -- significant quality gaps |
| 0-49 | Far from ready -- fundamental issues across multiple areas |

## Completion

Summarize:
- **Production readiness score: X/100** (test: A, review: B, security: C, lint: D, refactor: E)
- Number of commits made
- Key improvements made (from each skill)
- Test coverage status
- Security scan results
- Logging/observability status
- Skill improvements made (if any)
- Any remaining non-critical items for future consideration
