---
name: code-review
description: Perform a thorough code review looking for correctness bugs, SOLID violations, performance issues, and code smells. Use when the user asks to review code, check for issues, or improve code quality.
---

# Code Review Skill

Review changed code for correctness, clarity, and maintainability. Fix issues iteratively, most critical first. Principles are language-neutral; examples use Python for illustration.

## Scope

Review files changed on the current branch vs `main`:

```sh
git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only main...HEAD
```

If no branch diff exists (working on `main`), review recently modified files:

```sh
git diff --name-only HEAD~3
```

## Workflow

### Step 1: Understand Intent

Before judging code, understand what it's trying to do. Read commit messages, PR descriptions, and related tests. A reviewer who doesn't understand intent will flag correct tradeoffs as "issues."

```sh
# What changed and why?
git log --oneline main..HEAD
git diff main...HEAD --stat
```

For each changed file, answer: **What is this change trying to accomplish?** Then evaluate whether the implementation achieves that goal well.

### Step 2: First Pass -- Correctness

Read the diff looking for things that are **wrong**, not things that are ugly. Correctness issues trump all style concerns.

#### Bugs That Hide in Plain Sight

| Pattern | What to look for | Why it's dangerous |
|---|---|---|
| **Mutable default args** | `def f(items=[])`, `def f(config={})` | Shared across calls -- silent data corruption |
| **Mutating while iterating** | Modifying a collection while looping over it | Skips elements or raises errors |
| **Stale closure capture** | Lambda/callback capturing a loop variable | All closures share the final value |
| **Off-by-one in ranges** | Boundary conditions in loops and slices | Fencepost errors in any indexed logic |
| **Identity vs equality** | Using identity checks (`is`, `===`) for value comparison | Identity and equality are different things |
| **Swallowed exceptions** | `except: pass`, empty `catch {}` | Hides bugs; at minimum log the error |
| **Unguarded access** | Accessing a key/index without checking existence | Runtime errors on missing data |
| **Float equality** | `if distance == 3.0` | Float arithmetic is inexact -- use tolerance checks |
| **Shadowed builtins** | Naming a variable the same as a built-in function | Breaks stdlib functions silently |
| **Missing `return`** | Function returns a value on one path, falls through on another | Returns `null`/`None`/`nil` unexpectedly |

#### State and Data Integrity

- **Cache invalidation**: When cached data is computed, is the cache cleared when the source data changes? Look for memoized values, module-level dicts, LRU caches.
- **Shared mutable state**: If two functions modify the same data structure, can they interleave? Does the order matter?
- **Incomplete updates**: If a function updates A and B together, can it fail after updating A but before B?

### Step 3: Second Pass -- Structure

Now look at design quality. The goal is code that's easy to change next week, not code that's theoretically perfect.

#### SOLID -- What to Actually Look For

Don't check SOLID as a checklist. Instead, ask these diagnostic questions:

**Single Responsibility**: "If I needed to change the scoring logic, how many files would I touch?" If the answer is more than 1-2, responsibilities are scattered. If one file would need changes for 3 unrelated reasons, it has too many responsibilities.

**Open/Closed**: "Can I add a new state or variant without modifying the dispatch code?" Look for `if/elif/elif` chains on type or state -- these require modification for every new case. Dict dispatch or method-per-variant is open for extension.

**Liskov Substitution**: "If I swap one implementation for another, do callers break?" Check that subtypes don't: narrow input types, widen output types, add preconditions callers don't know about, or skip side effects the base type guarantees.

**Interface Segregation**: "Does this function/class depend on things it doesn't use?" A function that takes a large object but only uses one method could take a simpler interface. Look for parameters or attributes that are passed but never read.

**Dependency Inversion**: "Does high-level orchestration code import low-level implementation details?" Orchestration importing from internal helpers instead of going through the public module API is a violation.

#### Structural Smells

| Smell | Detection | Typical fix |
|---|---|---|
| **God class/module** | > 300 lines, multiple unrelated method clusters | Split into focused sub-modules |
| **Shotgun surgery** | One logical change requires edits in 5+ files | Consolidate related code |
| **Feature envy** | Method accesses another object's fields more than its own | Move method to the data's owner |
| **Long parameter list** | > 5 parameters | Group into a config object or data structure |
| **Deep nesting** | 4+ indent levels | Extract helper, use early returns |
| **Duplicate logic** | Same 5+ line pattern in 3+ places | Extract to shared function |
| **Primitive obsession** | Raw tuples/dicts passed everywhere for structured data | Introduce a type alias or data class |

### Step 4: Third Pass -- Performance (Hot Paths Only)

Only flag performance issues in code that runs frequently (inner loops, per-request handlers, hot paths). Don't optimize cold paths.

| Issue | Spot it | Fix |
|---|---|---|
| **O(n) lookup in a loop** | Linear search inside a loop | Convert to set/map before the loop |
| **Repeated computation** | Same expensive call with same args, multiple times | Cache or compute once and pass the result |
| **Unnecessary allocation** | Building a collection just to check length or existence | Use a lazy check or generator |
| **Construction in hot loop** | Rebuilding a data structure every iteration | Lift to outer scope if inputs don't change |
| **String concatenation in loop** | Appending strings one at a time in a loop | Use a builder or join pattern |

### Step 5: Fix and Verify

Process issues in severity order. For each fix:

1. **Fix** the issue (minimal change -- don't refactor adjacent code)
2. **Test**: Run the **Test (fast)** command from the CLAUDE.md Tooling table
3. **Lint**: Run the linter on changed files

Stop when no critical or major issues remain. Don't pursue minor style nits -- the linter handles those.

## Severity Classification

### Critical (must fix before merge)

These cause **incorrect behavior, data loss, or security holes**:

- Logic bug that produces wrong results
- Unsafe operation that could cause a runtime penalty or crash
- Cache that's never invalidated -- stale data across runs
- Unbounded loop/recursion without depth limit -- hang
- Exception swallowed silently -- bug hidden from detection
- Command injection, path traversal, or unsanitized external input

### Major (should fix)

These cause **maintenance burden or performance regression**:

- SOLID violation that makes the next change harder (god class, tight coupling)
- Missing type annotations on public API -- callers can't reason about contracts
- Performance issue on a hot path -- measurable impact
- Duplicated logic in 3+ places -- future changes will miss a copy
- Test that doesn't assert anything meaningful -- false confidence

### Minor (fix if convenient)

These are **style or clarity** improvements:

- Naming that could be clearer (but isn't misleading)
- Comment that restates the code
- Import ordering
- Opportunity to use a more idiomatic pattern

**Rule**: Never block a review on minor issues. Mention them if you fix them, don't chase them if you don't.

## Review Judgment Calls

Real reviews require judgment, not just rules. Guidelines for common gray areas:

**"Should I flag this style issue?"**
Only if it hurts readability for the *next* person. If two styles are equally clear, respect the existing convention in the file.

**"Is this abstraction premature?"**
If there's only one implementation and no concrete plan for a second, the abstraction is premature. Exception: if the abstraction makes testing dramatically easier.

**"Should I suggest a refactor?"**
Only if the current structure will make the *stated goal of this change* harder. Don't suggest refactors that serve hypothetical future changes.

**"This works but I'd do it differently."**
Not a review finding. Multiple correct approaches exist. Only flag it if your approach is measurably better (faster, clearer, more correct) -- not just different.

**"This violates a project convention."**
Check CLAUDE.md for the convention. If it's documented, flag it as major. If it's just your preference, skip it.

## Scoring (0-100)

Start at 100. Deduct points for each **unresolved** issue (after fixes in Step 5):

| Severity | Deduction per issue |
|----------|---------------------|
| Critical | -15 |
| Major | -5 |
| Minor | -1 |

| Score | Interpretation |
|-------|---------------|
| 90-100 | Production-ready -- no critical issues, minimal major issues |
| 70-89 | Needs attention -- some major issues remain |
| 50-69 | Significant issues -- multiple major or critical problems |
| 0-49 | Not ready -- critical issues unresolved |

## Gotchas

- **Over-flagging style as "major"**: Claude tends to classify naming and formatting issues as Major when they're Minor. Only flag style issues as Major if they actively mislead the reader.
- **Missing state mutation bugs**: Claude is better at spotting syntactic issues than semantic ones. Pay extra attention to cache invalidation, shared mutable state, and incomplete updates — these require reasoning about execution order, not just reading code.
- **Reviewing test code with production standards**: Test files have different conventions (longer functions, more repetition is OK). Don't flag test helpers for SRP violations.

## Output Format

Report findings grouped by severity, with file locations:

```
## Code Review: <branch or description>

### Critical
- `src/data/distance.py:45` -- Cache never cleared when source data changes; stale results after update

### Major
- `src/logic/planner.py:120` -- Method is 52 lines, exceeds 30-line limit; extract helper for route selection
- `src/logic/delivery.py:33` -- Missing type annotation on return value

### Minor
- `src/algorithms/search.py:88` -- Could use dict.get(key, default) instead of try/except KeyError

### Summary
- Files reviewed: 4
- Issues found: 1 critical, 2 major, 1 minor
- Issues fixed: 1 critical, 1 major
- Tests: all passing
- **Score: 89/100** (0 critical × -15, 1 major × -5, 1 minor × -1 remaining)
```
