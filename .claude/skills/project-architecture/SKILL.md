---
name: project-architecture
description: Use when deciding where to put code, choosing between patterns, designing module structure, or when user mentions architecture, code organization, or design patterns. Guides project architecture decisions including separation of concerns, module organization, and when to abstract; code examples use Python but principles are language-neutral.
---

# Project Architecture Patterns

> **Note**: Principles in this skill are language-neutral. Code examples use Python but the patterns (separate I/O from logic, functions over classes when stateless, etc.) apply to any language.

Evaluate and improve the structural design of code. This is a decision-making skill, not a catalog of patterns.

## When to Use

- User asks "where should this code go?"
- A module is growing past its size limit
- You need to choose between functions, classes, mixins, or protocols
- You're designing a new feature that touches multiple modules
- You're reviewing whether a decomposition is sound

## Workflow

### Step 1: Understand the Forces

Before proposing any structure, answer these questions:

1. **What changes together?** Group code by reason-to-change, not by technical layer.
2. **What are the dependencies?** Draw the import graph mentally. Cycles = structural problem.
3. **What needs to be tested independently?** If two things need different test setups, they belong in different modules.
4. **What is the performance contract?** Hot-path code has different structural constraints than cold-path code.

### Step 2: Evaluate Current Structure

Analyze the code using these metrics:

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| Module size | < 300 lines | > 500 lines, multiple unrelated responsibilities |
| Class size | < 200 lines | God class doing everything |
| Import depth | ≤ 3 levels (warning at 4, unhealthy at 5+) | Deep chains: a -> b -> c -> d -> e |
| Circular imports | Zero | Any |
| Public surface area | Small, intentional | Everything is public |
| Coupling | Modules share interfaces, not internals | Module A reaches into Module B's data structures |

If all metrics are in the healthy range, report that the architecture is sound and no restructuring is needed. This is a valid outcome — not every codebase needs refactoring. Note any minor opportunities but don't force changes.

### Step 3: Apply the Right Pattern

Use the decision frameworks below to choose the right structural pattern.

### Step 4: Validate the Design

After restructuring, verify:

- No circular imports (try importing the top-level package)
- Tests still pass (run the **Test (fast)** command from the CLAUDE.md Tooling table)
- No new lint issues (run linter on changed files)

## Decision Framework: Where Does This Code Go?

Don't ask "what kind of code is this?" -- ask "what forces act on it?"

```
Does it have state that persists across calls?
+-- No -> Function in the module closest to its callers
|       (If called from only one place, keep it there. Don't extract prematurely.)
|
+-- Yes -> Does it share that state across multiple methods?
    +-- No -> Function with explicit state parameter
    |       def compute(state: State, pos: Position) -> Action
    |
    +-- Yes -> Class. Now ask:
        +-- Is it data with minimal behavior? -> @dataclass / struct / record
        +-- Is it a long-lived object with complex state? -> Regular class
        +-- Is it a capability that cross-cuts multiple classes? -> Mixin / trait
```

## Decision Framework: When to Split a Module

A module should split when it has **multiple independent reasons to change**. Not when it's "big" -- a 280-line module with one cohesive responsibility is better than three 100-line modules with tangled dependencies.

**Split into a package when:**
- The module has 3+ distinct responsibilities with different change cadences
- Different parts need different test infrastructure
- You want to hide internal structure behind public re-exports

**How to split:**
```
# BEFORE: data_state.py (500 lines, 4 responsibilities)

# AFTER: data_state/ package
data_state/
+-- __init__.py          # Re-exports public API
+-- _base.py             # Shared helpers, base class
+-- distance.py          # Distance computation
+-- assignment.py        # Entity-to-target assignment
+-- routing.py           # Multi-stop route optimization
```

Key rules:
- The public entry point re-exports the public API -- callers don't change
- Internal modules can use `_prefix` to signal they're private
- Each sub-module should be testable independently

## Decision Framework: Composition Patterns

### Functions (Default Choice)

Use when: no shared state, no configuration, pure input -> output.

```python
def find_shortest_path(start: Pos, goal: Pos, blocked: set[Pos]) -> int | None:
    """Stateless. Easy to test. Easy to compose."""
```

### Classes

Use when: shared mutable state across multiple methods, or when identity matters.

```python
class DataState:
    """Owns the data grid, caches, and per-round state.
    Multiple methods operate on the same internal data."""

    def __init__(self, width: int, height: int) -> None:
        self._grid = Grid(width, height)
        self._cache: dict[Pos, int] = {}

    def update(self, new_data: dict) -> None: ...
    def distance(self, a: Pos, b: Pos) -> int: ...
```

### Mixins

Use when: a class has grown large and its methods partition into independent capabilities that share `self`.

```python
class DistanceMixin:
    """Distance computation. Mixed into DataState."""

    def dist_static(self, a: Pos, b: Pos) -> int:
        # Uses self._grid, self._cache from the host class
        ...

class DataState(DistanceMixin, RoutingMixin, AssignmentMixin):
    """Composes capabilities via mixins."""
```

**Mixin rules:**
- Each mixin has one responsibility
- Mixins access only well-defined attributes of `self` (document which ones)
- Mixins don't depend on each other -- they depend on the base class
- The host class is the only place where mixins are composed
- Test each mixin through the composed class, not in isolation

### Protocols (Structural Subtyping)

Use when: you need polymorphism without inheritance. When multiple unrelated classes should satisfy the same interface.

```python
from typing import Protocol

class Simulator(Protocol):
    def step(self, actions: list[Action]) -> Result: ...
    def is_done(self) -> bool: ...

# LiveSimulator and ReplaySimulator both satisfy this
# without inheriting from a common base
```

**Protocol vs ABC decision:**
- Use Protocol when: consumers just need "anything with these methods" (duck typing made explicit)
- Use ABC when: you have shared implementation in the base class that subclasses inherit

## Dependency Direction

Dependencies should flow **inward** -- from I/O and entry points toward pure domain logic. Never the reverse.

```
Entry points (main.py, cli.py)
    | depends on
Orchestration (logic/, handlers/)
    | depends on
Domain logic (core/, algorithms/)
    | depends on
Pure data (constants, dataclasses)
```

**Rules:**
- Lower layers NEVER import from higher layers
- Constants/config imports nothing from the project
- Core algorithms import only from constants (or stdlib)
- Domain logic imports from core -- never the reverse
- Entry points import from everything -- nothing imports from entry points

**Detecting violations:**
```sh
# Check what a module imports from the project
grep -n "from src" src/algorithms/search.py
# If algorithms imports from logic/ -> dependency direction violation
```

## Module Cohesion Checklist

When reviewing a module, check:

- [ ] **Can you describe what it does in one sentence without "and"?** If not, it has multiple responsibilities.
- [ ] **Do all public functions/methods use roughly the same set of internal helpers?** If some functions use helpers A, B, C and others use D, E, F -- those are two modules.
- [ ] **Would a change to one function likely require changes to others in the same file?** High co-change = high cohesion = good.
- [ ] **Can you delete a public function without breaking anything else in the file?** If yes for many functions, cohesion is low -- it's a grab bag.

## Anti-Patterns and Remedies

| Anti-Pattern | Symptom | Remedy |
|---|---|---|
| **God class** | One class > 500 lines, does everything | Extract mixins or delegate to helper classes |
| **Shotgun surgery** | One change requires editing 5+ files | The scattered code belongs together -- consolidate |
| **Feature envy** | Method mostly accesses another object's data | Move the method to where the data lives |
| **Middle man** | Class that only delegates to another class | Remove it; let callers use the real thing directly |
| **Speculative generality** | ABC with one implementation, unused Protocol | Delete it. Add abstraction when the second use case arrives |
| **Circular dependency** | A imports B, B imports A | Extract shared types/interfaces into a third module, or merge |
| **Util junk drawer** | `utils` > 200 lines with unrelated functions | Split by domain or move functions to their callers |

## Scaling Patterns

### When a function grows complex (> 30 lines)

Extract named helper functions in the same module. Don't create a new file for one helper.

### When a module grows large (> 300 lines)

1. Identify the 2-3 responsibilities
2. Check if they have different test needs
3. If yes: split into a package with public re-exports
4. If no: try extracting just the helper functions to a `_helpers` module

### When a class grows large (> 200 lines)

1. Identify method clusters that share internal state
2. Extract each cluster as a mixin
3. Keep initialization and coordination in the main class
4. Each mixin gets tested through the composed class

### When you need a new top-level module

Justify it: Does it have a distinct responsibility? Will multiple other modules import from it? If it's only used by one module, make it a private sub-module instead.

## Common Architectural Decisions

### State management: class instance vs module-level

| Factor | Class instance | Module-level |
|--------|---------------|--------------|
| Multiple instances needed | Yes -> class | No -> either works |
| Needs reset between runs | Class with `reset()` method | Module `reset()` function |
| Accessed from many modules | Class passed as parameter | Import the module |
| Performance-critical cache | Class attribute (co-located) | Module dict (slightly faster) |

### Configuration: constants vs config objects

- **Constants** (`ALL_CAPS` in constants file): values that change at development time, not runtime. Thresholds, sizes, feature flags.
- **Config dataclass**: values that change per-run or per-environment. Passed explicitly, never imported as globals.

### Error handling location

- **System boundaries** (I/O, user input, external APIs): validate and handle errors
- **Internal pure functions**: let exceptions propagate. Don't add try/except inside domain logic -- the caller decides how to handle failure.
- **Between layers**: convert low-level exceptions to domain-meaningful ones at layer boundaries

## Gotchas

- **Recommending patterns the team doesn't know**: Introducing hexagonal architecture to a team that writes simple CRUD apps adds complexity without benefit. Match the architecture to the team's capabilities and the project's actual complexity.
- **Over-modularizing small projects**: A 5-file project doesn't need 3 abstraction layers. Start simple, split when you feel pain (files over 300 lines, too many reasons to change one module).
- **Splitting by technical layer when the domain says otherwise**: Organizing as `models/`, `services/`, `controllers/` scatters related code across directories. For domain-heavy apps, organize by feature (`auth/`, `payments/`, `notifications/`).

## Related Skills

| Need | Skill |
|------|-------|
| Restructuring existing code | `refactor` |
| Evaluating code quality | `code-review` |
| Adding caching to hot paths | `caching-strategies` |
| Processing pipeline design | `data-pipeline` |
| Error handling patterns | `error-handling` |
