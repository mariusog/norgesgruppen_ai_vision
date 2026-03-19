---
name: tdd-cycle
description: Use when the user wants to implement a feature using TDD, write tests first, follow test-driven practices, or mentions red-green-refactor. Guides the Red-Green-Refactor cycle; examples use Python/pytest but principles are language-neutral.
---

# TDD Cycle Skill

## Overview

This skill guides you through the Test-Driven Development cycle:
1. **RED**: Write a failing test that describes desired behavior
2. **GREEN**: Write minimal code to pass the test
3. **REFACTOR**: Improve code while keeping tests green

## Workflow Checklist

```
TDD Progress:
- [ ] Step 1: Understand the requirement
- [ ] Step 2: Choose test type (unit/integration/e2e)
- [ ] Step 3: Write failing test (RED)
- [ ] Step 4: Verify test fails correctly
- [ ] Step 5: Implement minimal code (GREEN)
- [ ] Step 6: Verify test passes
- [ ] Step 7: Refactor if needed
- [ ] Step 8: Verify tests still pass
```

## Step 1: Requirement Analysis

Before writing any code, understand:
- What is the expected input?
- What is the expected output/behavior?
- What are the edge cases?
- What errors should be handled?
- Is the behavior deterministic? If not, how is the seed handled?

## Step 2: Choose Test Type

| Test Type | Use For | Example |
|-----------|---------|---------|
| Unit test | Pure functions, class methods | Testing `calculate_score()` |
| Integration test | Module interactions, pipelines | Testing full decision flow |
| Parametrized test | Same logic, many inputs | Testing algorithm edge cases |
| Regression test | Preventing known bugs | Testing fixed issue doesn't recur |

## Step 3: Write Failing Test (RED)

### Test Structure

```python
import pytest
from module_under_test import function_or_class


class TestClassName:
    """Tests for ClassName behavior."""

    def test_expected_behavior(self):
        result = function_or_class(input_value)
        assert result == expected_value

    def test_edge_case(self):
        with pytest.raises(ValueError, match="specific message"):
            function_or_class(invalid_input)

    @pytest.mark.parametrize("input_val,expected", [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ])
    def test_multiple_cases(self, input_val, expected):
        assert function_or_class(input_val) == expected
```

### Good Test Characteristics
- **One behavior per test**: Each test function tests one thing
- **Clear naming**: `test_<what>_<condition>_<expected>` pattern
- **Minimal setup**: Only create data needed for the specific test
- **Fast execution**: Mock external dependencies, avoid I/O
- **Independent**: Tests don't depend on order or shared mutable state
- **Deterministic**: Fixed seeds for any randomized behavior

### Test Data
Use fixtures and factory functions:
```python
@pytest.fixture
def sample_state():
    """Build a minimal state for testing."""
    return make_state(
        items=[{"id": "a", "type": "widget", "position": [1, 1]}],
        config={"seed": 42},
    )
```

## Step 4: Verify Failure

Run the new test using the **Test (debug)** command from the CLAUDE.md Tooling table.

The test MUST fail. If it passes immediately:
- The behavior already exists
- The test is wrong (not testing what you think)

## Step 5: Implement (GREEN)

Write the MINIMUM code to pass:
- No optimization yet
- No edge case handling (unless that's what you're testing)
- Just make it work

## Step 6: Verify Pass

Run the test again using the **Test (debug)** command.

## Step 7: Refactor

Improve the code while keeping tests green:
1. Make ONE change at a time
2. Run tests after EACH change
3. If tests fail, undo and try a different approach
4. Stop when code is clean (don't over-engineer)

## Step 8: Final Verification

Run the **Test (fast)** command from the CLAUDE.md Tooling table. All tests must pass.

## Anti-Patterns to Avoid

1. **Testing implementation, not behavior**: Test what it does, not how
2. **Too many assertions**: Split into separate test functions
3. **Brittle tests**: Don't test exact error messages or timestamps
4. **Slow tests**: Mock external services, avoid unnecessary I/O
5. **Mystery data**: Make test data explicit and visible in each test
6. **Non-deterministic tests**: Always use fixed seeds for randomized behavior

## Gotchas

- **Writing tests after implementation and calling it TDD**: TDD means the test exists BEFORE the code. If you write the code first, you're writing regression tests — which is fine, but it's not TDD and you lose the design benefits.
- **Tests that are too coupled to implementation**: Asserting on internal variable names, private method calls, or exact data structures makes tests break on every refactor. Test observable behavior through the public API.
- **Skipping the refactor step**: Green does not mean done. After the test passes, refactor the implementation for clarity. The Red-Green-Refactor cycle has three steps, not two.
