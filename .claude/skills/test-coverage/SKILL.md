---
name: test-coverage
description: Ensure adequate test coverage for changed code. Identifies relevant tests, runs them, and checks coverage. Use when the user asks to check test coverage, run tests, or verify specs.
---

# Test Coverage Skill

Ensure changed code has adequate test coverage and all tests pass.

## Scope

This skill applies to files changed in the current branch compared to `main` (or `origin/main`).
Identify changed files with: `git diff --name-only origin/main...HEAD`

## Step 1: Identify Relevant Tests

For each changed file, find corresponding test files:

### Mapping Source to Tests

Source and test directories should mirror each other. For example:
- `src/module` -> `tests/test_module`
- `src/package/module` -> `tests/package/test_module`

Use grep to find test references to your module in the test directory.

## Step 2: Baseline Test Run

Run identified tests using the **Test (fast)** command from the CLAUDE.md Tooling table to ensure we start green.

If tests fail:
- **STOP** -- Do not proceed with other quality steps
- Report failing tests to the user
- Broken tests must be fixed before continuing

## Step 3: Coverage Analysis

For each changed file, verify test coverage:

### Check for Missing Tests
- Does every new or significantly modified public function have a corresponding test? ("Modified" means changed logic, parameters, or return values -- not just formatting or comment edits.)
- For modified functions that already have tests, check whether the change introduces new code paths that need additional test scenarios.
- Are new classes covered by tests?
- Are new algorithms tested with representative inputs?

### Check for Missing Scenarios
For existing tests, verify coverage of:
- **Happy path** -- normal operation
- **Edge cases** -- empty inputs, boundary values, None/zero
- **Error paths** -- invalid inputs, unreachable goals, timeouts
- **State transitions** -- before/after mutations
- **Determinism** -- same input always produces same output

### Add Missing Tests
If coverage is insufficient:
1. Write unit tests for new/changed public functions
2. Write integration tests for new workflows
3. Add edge case and error path tests
4. Use parameterized tests for multiple input scenarios

### Coverage Tool

Use your language's coverage tool to identify untested code paths. Check the CLAUDE.md Tooling table for the project-specific command.

If no coverage tool is configured, manually verify coverage by reading each changed public function and confirming a corresponding test exists in the test directory.

## Step 4: Run Full Test Suite

Run the **Test (fast)** command from the CLAUDE.md Tooling table. Ensure all tests pass before completing.

## Step 5: Verify No Regressions

If refactoring was done alongside coverage work:
- Run the full test suite to catch regressions
- Check that no existing tests were broken
- Verify benchmark scores haven't degraded (if applicable)

## Scoring (0-100)

Score across two dimensions:

### Coverage Breadth (50 points)

| Criteria | Points |
|----------|--------|
| Every public function/method has at least one test | 20 |
| Every new class/module has test coverage | 10 |
| Integration tests exist for cross-module workflows | 10 |
| Parameterized tests cover multiple input scenarios | 10 |

### Test Quality (50 points)

| Criteria | Points |
|----------|--------|
| Happy path covered for each tested function | 10 |
| Edge cases covered (empty, zero, boundary) | 10 |
| Error paths covered (invalid input, failures) | 10 |
| All tests are deterministic (fixed seeds where needed) | 10 |
| Test names follow `test_<method>_<scenario>` convention | 5 |
| Arrange-Act-Assert pattern used consistently | 5 |

| Score | Interpretation |
|-------|---------------|
| 90-100 | Thorough -- all public methods tested with quality scenarios |
| 70-89 | Adequate -- most methods tested, some gaps in edge cases |
| 50-69 | Incomplete -- significant coverage or quality gaps |
| 0-49 | Insufficient -- major public methods lack tests |

## Gotchas

- **Testing implementation details**: Writing tests that assert on internal variable names, private method calls, or specific data structures instead of observable behavior. These tests break on every refactor.
- **Ignoring edge cases for "simple" functions**: Even a function that returns a boolean needs tests for null input, empty collections, and boundary values.
- **Copy-paste test bodies**: Duplicating test logic across scenarios instead of using parameterized tests. This leads to tests that silently drift out of sync.

## Completion

Report:
- Number of test files identified
- Number of tests run
- Pass/fail status
- Any new tests added
- Coverage gaps that still need attention
- **Score: X/100** (breadth: Y/50, quality: Z/50)
