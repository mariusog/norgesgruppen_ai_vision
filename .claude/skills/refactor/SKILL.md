---
name: refactor
description: Refactor code for clarity and maintainability. Includes control flow improvements, constant extraction, and comment cleanup. Use when the user asks to refactor code, clean up code, or improve code structure.
---

# Refactor Skill

Improve code clarity and maintainability through targeted refactoring. Principles are language-neutral; examples use Python for illustration.

## Scope

This skill applies to files changed in the current branch compared to `main` (or `origin/main`).
Identify changed files with: `git diff --name-only origin/main...HEAD`

## Step 1: Control Flow Refactoring

### Replace Conditionals with Dictionaries
```python
# Before
def status_color(status):
    if status == "success":
        return "green"
    elif status == "warning":
        return "yellow"
    elif status == "error":
        return "red"
    else:
        return "gray"

# After
STATUS_COLORS = {"success": "green", "warning": "yellow", "error": "red"}

def status_color(status):
    return STATUS_COLORS.get(status, "gray")
```

### Use Early Returns
```python
# Before
def process(data):
    if data is not None:
        if data.is_valid():
            # long processing logic
            pass

# After
def process(data):
    if data is None:
        return
    if not data.is_valid():
        return
    # long processing logic
```

### Simplify with Language Idioms
```python
# Before
result = []
for item in items:
    if item.is_active():
        result.append(item.name)

# After
result = [item.name for item in items if item.is_active()]
```

## Step 2: Extract Constants

### Magic Numbers
```python
# Before
def calculate_score(value):
    return value * 1.15 + 50

# After
SCORE_MULTIPLIER = 1.15
BASE_SCORE_BONUS = 50

def calculate_score(value):
    return value * SCORE_MULTIPLIER + BASE_SCORE_BONUS
```

### Placement Guidelines
- Module-level constants: ALL_CAPS at the top of the module
- Shared constants: Dedicated constants module (see **Constants file** in the CLAUDE.md Tooling table)
- Configuration values: Environment variables or config files

## Step 3: Comment Cleanup

### Remove Unnecessary Comments
Delete comments that:
- Simply describe what the code does (code should be self-documenting)
- Restate the function name or variable name
- Are outdated or no longer accurate
- Are commented-out code

### Keep Valuable Comments
Preserve comments that:
- Explain WHY something is done a certain way
- Document non-obvious business logic or algorithm choices
- Warn about edge cases or gotchas
- Reference external documentation or tickets

## Step 4: File and Naming Conventions

Follow your language's standard naming conventions. Common principles:
- **Modules/files**: lowercase, consistent with language convention
- **Classes/types**: PascalCase (most languages)
- **Functions/variables**: camelCase or snake_case (per language convention)
- **Constants**: UPPER_SNAKE_CASE (most languages)
- **Private/internal**: use the language's visibility mechanism (underscore prefix, `private` keyword, unexported names, etc.)
- **Test files**: follow the test framework's naming convention

## Step 5: Verify Changes

After refactoring, use the commands from the CLAUDE.md Tooling table:
1. Run **Test (fast)** to confirm no regressions
2. Run **Lint** to confirm no style issues

If tests fail after refactoring:
1. Identify whether the refactor changed behavior (bug introduced) or exposed a pre-existing test issue
2. If the refactor changed behavior: revert the specific refactor that broke the test and try a different approach
3. If the test was brittle (testing implementation details rather than behavior): fix the test to assert on behavior
4. Never commit with failing tests

## Gotchas

- **Changing behavior during refactoring**: Refactoring must preserve behavior. If a test fails after refactoring, the refactor changed something it shouldn't have — revert and try again.
- **Over-extracting constants**: Not every number needs to be a named constant. `range(len(items))` doesn't need `ITEM_RANGE_START = 0`. Extract only values that represent tunable thresholds or business rules.
- **Removing "unnecessary" comments that explain WHY**: Comments that restate WHAT the code does should go. Comments that explain WHY a non-obvious choice was made must stay.

## Scoring (0-100)

Evaluate code clarity across five dimensions. Award up to 20 points each:

| Dimension | 0-5 (Poor) | 6-10 (Needs work) | 11-15 (Good) | 16-20 (Excellent) |
|-----------|-----------|-------------------|--------------|-------------------|
| **Control flow** | Deep nesting, long if/elif chains | Some nesting, few early returns | Mostly flat, guard clauses used | Flat flow, dict dispatch where appropriate |
| **Constants** | Magic numbers throughout | Some extracted, some inline | Most in named constants | All thresholds/config in constants file |
| **Comments** | Restating code or outdated | Mix of useful and noise | Mostly "why" comments | Only valuable comments remain |
| **Naming** | Misleading or single-letter | Some unclear names | Clear and descriptive | Self-documenting, follows conventions |
| **File structure** | Files >300 lines, mixed concerns | Some large files | Within limits, clear organization | Focused modules, single responsibility |

Score each dimension independently, then sum.

| Score | Interpretation |
|-------|---------------|
| 90-100 | Clean -- code is clear and well-structured |
| 70-89 | Good -- minor clarity improvements possible |
| 50-69 | Needs refactoring -- multiple dimensions below standard |
| 0-49 | Poor -- significant structural and clarity issues |

## Completion

Report:
- Number of control flow improvements made
- Number of constants extracted
- Number of comments cleaned up
- Any areas that could benefit from further refactoring
- **Score: X/100** (breakdown: control flow Y/20, constants Y/20, comments Y/20, naming Y/20, structure Y/20)
