---
name: lint
description: Run linter, formatter, and type checker to fix code style and quality issues. Use when the user asks to lint code, fix style issues, or run code quality checks.
---

# Lint Skill

Fix code style and quality issues using the project's configured linter, formatter, and type checker.

## Scope

This skill applies to files changed in the current branch compared to `main` (or `origin/main`).
Identify changed files with: `git diff --name-only origin/main...HEAD`

## Principles (language-neutral)

1. Fix all linter warnings -- do NOT suppress them with ignore comments
2. Format all changed files consistently
3. Fix type errors with real type annotations, not suppression comments
4. Do NOT disable any rules
5. Do NOT remove any test functions

## Step 1: Identify Changed Files

Categorize by file type. Only lint files in the project's source language.

## Step 2: Lint and Format

Use the **Lint** and **Format** commands from the CLAUDE.md Tooling table.

### Python (default)

```bash
ruff check <changed files>            # Check
ruff check --fix <changed files>      # Auto-fix
ruff format <changed files>           # Format
```

Common rules: F (unused imports), E/W (style), I (import order), UP (modernize syntax), B (common bugs), SIM (simplifiable code).

### TypeScript/JavaScript

```bash
npx eslint <changed files>            # Check
npx eslint --fix <changed files>      # Auto-fix
npx prettier --write <changed files>  # Format
```

### Go

```bash
golangci-lint run                     # Check
gofmt -w <changed files>             # Format
```

## Step 3: Type Check

Use the **Type check** command from the CLAUDE.md Tooling table.

Fix type errors properly:
- Add missing type annotations
- Fix incompatible types
- Do NOT suppress with ignore comments unless truly necessary (document why)

## Step 4: Verify Clean

Run linter and formatter one final time to confirm all issues are resolved.

## Gotchas

- **Auto-fix removing re-exported imports**: Ruff's auto-fix may remove imports that look unused but are re-exported for external consumers (e.g., `from .module import SomeClass` in `__init__.py`). Verify before accepting.
- **Suppressing instead of fixing**: When a lint rule is hard to satisfy, the temptation is to add an ignore comment. Fix the code instead — ignore comments accumulate and erode lint value over time.
- **Type annotations that lie**: Adding `-> None` to a function that sometimes returns a value, or `str` to a parameter that also accepts `None`. Wrong annotations are worse than missing ones.

## Scoring (0-100)

Start at 100. Deduct points for each **unresolved** offense (after auto-fix and manual fixes):

| Category | Deduction per offense |
|----------|----------------------|
| Type error | -5 |
| Lint error (bug-risk rules) | -3 |
| Lint warning (style rules) | -1 |
| Format violation | -1 |

| Score | Interpretation |
|-------|---------------|
| 90-100 | Clean -- no type errors, minimal lint warnings |
| 70-89 | Acceptable -- some warnings, no errors |
| 50-69 | Messy -- multiple errors or type issues |
| 0-49 | Needs work -- pervasive issues across files |

## Completion

Report:
- Number of files linted
- Number of offenses fixed
- Any remaining issues that need manual attention
- **Score: X/100** (deductions itemized by category)
