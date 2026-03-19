---
name: debugging
description: Systematic debugging workflow for isolating and fixing bugs. Use when tests fail unexpectedly, behavior doesn't match expectations, a user reports a bug, or when an agent encounters an error it doesn't immediately understand.
---

# Debugging Skill

## Overview

Debugging is a systematic process, not guesswork. Follow this workflow every time. Do NOT change code until you understand the root cause. Principles are language-neutral; examples use Python for illustration.

## Step 1: Reproduce

Before anything else, get a reliable reproduction.

If a test fails, rerun just that test with more detail using the **Test (debug)** command from the CLAUDE.md Tooling table.

If it's a runtime bug, find or write a minimal reproducing test:
- Same seed
- Same config
- Smallest possible input that triggers the bug

**Key questions:**
- Can you reproduce it every time? (If no, it's a non-determinism bug -- check seeds)
- What's the minimal input that triggers it?
- Does it reproduce with a clean state (caches cleared, fresh fixtures)?

**If you can't reproduce it, STOP.** A bug you can't reproduce is a bug you can't verify as fixed. Investigate the non-determinism first (missing seed, shared mutable state, test interdependence).

## Step 2: Isolate

Narrow down WHERE the bug is. Work from the error backward.

### Read the Error

Run the failing test with the **Test (debug)** command from the CLAUDE.md Tooling table to get the full error output.

The error output tells you:
- **Which line** raised the error
- **Which function** called it
- **What values** were involved (in the assertion message)

### Binary Search the Cause

If the error isn't obvious from the traceback:

Stop as soon as you identify the cause. These checks are ordered by likelihood -- you rarely need all four. Once a check reveals the problem, proceed to Step 3.

1. **Check the input**: Is the test data valid? Positions within bounds? Types correct?
2. **Check the function under test**: Add a focused test with the simplest possible input
3. **Check dependencies**: Does the function depend on state that wasn't set up correctly?
4. **Check recent changes**: `git diff` -- did a recent change break this?

```sh
# What changed recently?
git log --oneline -10
git diff HEAD~1 -- src/
```

### Isolate with Print (temporarily)

If you need to see intermediate values, use a temporary debug test -- NOT print statements in source code:

```python
def test_debug_specific_case():
    """Temporary: isolate the bug. Delete after fixing."""
    state = make_state(...)  # minimal reproduction
    # Call sub-functions individually to find where it breaks
    result_a = step_a(state)
    assert result_a == expected_a, f"step_a returned {result_a}"

    result_b = step_b(result_a)
    assert result_b == expected_b, f"step_b returned {result_b}"
```

## Step 3: Understand the Root Cause

Before writing a fix, state the root cause in one sentence:

> "The bug is caused by [X] because [Y], which results in [Z]."

Examples:
- "The cache returns stale data because it's not cleared between rounds, which causes the distance to be wrong for moved entities."
- "The function crashes on empty input because the min() call has no default, which raises ValueError when the list is empty."

If you can't state the root cause clearly, you haven't finished isolating.

## Step 4: Write a Regression Test

Write a test that FAILS with the current bug and will PASS after the fix. This ensures the bug never comes back.

```python
def test_function_handles_empty_input():
    """Regression: previously crashed with ValueError on empty list."""
    result = function_under_test([])
    assert result is None  # Or whatever the correct behavior should be
```

Run it to confirm it fails using the **Test (debug)** command from the CLAUDE.md Tooling table.

## Step 5: Fix

Now -- and only now -- fix the code.

Rules:
- **Minimal fix**: Change the fewest lines possible to fix the bug
- **Don't refactor while fixing**: Fix the bug, commit, THEN refactor (if needed)
- **Don't fix adjacent issues**: One bug per fix. File separate tasks for other problems.
- **Fix the root cause, not the symptom**: If the cache is stale, fix the invalidation -- don't add a workaround that re-fetches.

## Step 6: Verify

1. Run the regression test with the **Test (debug)** command -- confirm it passes
2. Run the full suite with the **Test (fast)** command -- confirm no regressions
3. If benchmark-relevant, check `docs/benchmark_results.md` for regressions

## Step 7: Report

Update TASKS.md with:
```
Result: Fixed <root cause> | added regression test | tests: N pass
```

## Common Bug Patterns

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| Test passes alone, fails in suite | Shared mutable state between tests | Check autouse fixtures, module-level caches |
| Different results each run | Missing seed, set iteration order, dict ordering | Search for `random`, unordered `set` iteration |
| Works for small input, fails for large | Off-by-one, unbounded iteration, memory | Check boundary conditions, max_steps limits |
| Correct on first call, wrong on second | Stale cache, mutation of shared data | Check cache invalidation, defensive copies |
| Assertion error with correct-looking values | Float comparison, type mismatch, off-by-one | Use approximate comparison for floats, check types |
| Import error or attribute error | Circular import, renamed function, missing module export | Check import chain, recent renames |

## Gotchas

- **Fixing the symptom instead of the root cause**: Adding a null check to suppress a crash instead of understanding why the value is null. Always ask "why is this state reachable?" before adding a guard.
- **Changing multiple things at once**: When debugging, change ONE variable at a time. If you change the input AND the function AND the test simultaneously, you can't isolate which change fixed (or broke) things.
- **Assuming the bug is in the code you're looking at**: The actual bug is often in the caller, the setup, or the test fixture — not the function under investigation.

## Anti-Patterns

- **Shotgun debugging**: Changing random things and rerunning. Follow the workflow.
- **Fixing the test instead of the code**: If the test caught a real bug, fix the code.
- **Adding workarounds**: Fix root causes. `try/except: pass` is not a fix.
- **Debugging without reproduction**: If you can't reproduce it, you can't verify the fix.
- **Skipping the regression test**: Every bug fix needs a test that prevents recurrence.
