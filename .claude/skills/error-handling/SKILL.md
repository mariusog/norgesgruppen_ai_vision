---
name: error-handling
description: Use when adding error handling, designing exception hierarchies, fixing swallowed errors, or when user mentions errors, exceptions, validation, or error recovery. Covers custom exceptions, error propagation, boundary validation, and recovery; examples use Python but principles are language-neutral.
---

# Error Handling Patterns

> **Note**: Error handling principles (fail fast, don't swallow, validate at boundaries) apply to any language. Code examples use Python.

## Core Principles

1. **Fail fast** -- detect errors early, close to the source
2. **Don't swallow errors** -- never use bare `except:` / `catch` without re-raising or logging
3. **Validate at boundaries** -- check external input at system edges, trust internal code
4. **Use specific exceptions** -- catch the narrowest type possible
5. **Errors are data** -- return structured error info, not just strings

## Where to Validate (and Where NOT to)

| Location | Validate? | Example |
|----------|-----------|---------|
| External input (user, API, file) | YES | Check types, ranges, required fields |
| Module public API | YES | Validate arguments on public functions |
| Internal private functions | NO | Trust callers within the same module |
| Between your own modules | MINIMAL | Assert preconditions in debug mode |
| Framework callbacks | YES | Validate data from framework before processing |

## Custom Exception Hierarchy

Define project-specific exceptions. Keep the hierarchy shallow (2 levels max: one base exception, then specific exceptions that inherit from it). You can have many specific exceptions at the same level — the constraint is depth, not breadth.

```python
class ProjectError(Exception):
    """Base exception for this project. All custom errors inherit from this."""

class ConfigError(ProjectError):
    """Invalid configuration or missing required config."""

class ValidationError(ProjectError):
    """Input fails validation rules."""

class ResourceNotFoundError(ProjectError):
    """Expected resource (file, entity, record) does not exist."""

class BudgetExceededError(ProjectError):
    """Operation exceeded its iteration/time/memory budget."""
```

### Why Custom Exceptions Matter

```python
# BAD: Caller can't distinguish error types
def load_config(path):
    if not os.path.exists(path):
        raise Exception("Config not found")  # Generic -- useless for callers
    data = json.loads(open(path).read())
    if "seed" not in data:
        raise Exception("Missing seed")  # Same type for different problems

# GOOD: Caller can handle each case differently
def load_config(path):
    if not os.path.exists(path):
        raise ResourceNotFoundError(f"Config file missing: {path}")
    data = json.loads(open(path).read())
    if "seed" not in data:
        raise ConfigError("Required key 'seed' missing from config")
```

## Error Propagation Rules

### Let Errors Bubble Up

```python
# BAD: Swallowing the error
def get_score(entity_id):
    try:
        return compute_score(entity_id)
    except Exception:
        return 0  # Silently returns wrong data -- bugs hide here

# GOOD: Let it propagate (caller decides how to handle)
def get_score(entity_id):
    return compute_score(entity_id)

# GOOD: Catch specific, add context, re-raise
def get_score(entity_id):
    try:
        return compute_score(entity_id)
    except KeyError as e:
        raise ResourceNotFoundError(f"Entity {entity_id} not found") from e
```

### The `from e` Pattern

Always chain exceptions so the original traceback is preserved:

```python
try:
    result = parse(raw_data)
except json.JSONDecodeError as e:
    raise ValidationError(f"Invalid data format: {raw_data[:50]}") from e
```

## Boundary Validation

Validate at the system edge, then pass clean data inward.

```python
def handle_request(raw_input: dict) -> dict:
    """System boundary -- validate everything here."""
    config = validate_config(raw_input.get("config", {}))
    entities = validate_entities(raw_input.get("entities", []))
    # From here inward, all functions trust their inputs
    return process(config, entities)

def validate_config(raw: dict) -> Config:
    """Validate and convert raw dict to typed Config."""
    required = ["seed", "max_steps"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")
    if not isinstance(raw["seed"], int):
        raise ConfigError(f"seed must be int, got {type(raw['seed']).__name__}")
    if raw["max_steps"] <= 0:
        raise ConfigError(f"max_steps must be positive, got {raw['max_steps']}")
    return Config(**raw)

def validate_entities(raw: list) -> list[Entity]:
    """Validate entity data at the boundary."""
    if not raw:
        raise ValidationError("Entity list cannot be empty")
    entities = []
    for i, e in enumerate(raw):
        if "id" not in e or "position" not in e:
            raise ValidationError(f"Entity {i} missing required fields")
        entities.append(Entity(id=e["id"], position=tuple(e["position"])))
    return entities
```

## Bounded Operations

All loops and recursive calls must have explicit limits:

```python
from constants import MAX_SEARCH_STEPS

def search(start, goal, grid):
    """BFS with bounded iteration."""
    queue = deque([start])
    visited = {start}
    steps = 0
    while queue:
        steps += 1
        if steps > MAX_SEARCH_STEPS:
            raise BudgetExceededError(
                f"Search exceeded {MAX_SEARCH_STEPS} steps from {start} to {goal}"
            )
        current = queue.popleft()
        if current == goal:
            return reconstruct_path(current)
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return None  # No path exists
```

## Recovery Patterns

### Choosing a Recovery Strategy

- Error is **transient** (network timeout, rate limit, temporary unavailability) → **Retry** with exponential backoff
- Error is **permanent** but a degraded result is acceptable (cache miss, optional feature unavailable) → **Fallback** to a default or cached value
- Error is **permanent** and no degraded result is acceptable (missing required data, auth failure) → **Propagate** the error to the caller
- Error could be either → **Retry once**, then **fallback** if retry fails

### Retry with Backoff (for transient failures only)

```python
import time

def retry(fn, max_attempts=3, backoff=1.0):
    """Retry a function on transient errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_attempts:
                raise  # Exhausted retries -- propagate
            time.sleep(backoff * attempt)
```

### Fallback (when degraded output is acceptable)

```python
def get_optimal_path(start, goal, grid):
    """Try optimal path, fall back to greedy if budget exceeded."""
    try:
        return astar_search(start, goal, grid)
    except BudgetExceededError:
        logger.warning("A* exceeded budget, falling back to greedy")
        return greedy_search(start, goal, grid)
```

## Logging Errors

```python
import logging
logger = logging.getLogger(__name__)

# BAD: Logging and raising (causes duplicate logs up the chain)
try:
    result = process(data)
except ProcessError as e:
    logger.error(f"Failed: {e}")
    raise  # Caller will also log it

# GOOD: Log at the handler, not at every level
def top_level_handler(data):
    try:
        return process(data)
    except ProjectError as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return error_response(e)
```

## Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Bare `except:` / `catch(Exception)` | Catch specific exception types |
| `except: pass` (swallowed error) | Log, re-raise, or return error value |
| Returning `None` on error without docs | Raise an exception, or use `Optional` with clear docs |
| Validating deep inside internal code | Move validation to the boundary |
| Catching and returning generic error string | Use typed exceptions with structured data |
| Retry on non-transient errors | Only retry `ConnectionError`, `TimeoutError`, etc. |
| Logging at every catch level | Log once at the top-level handler |

## Gotchas

- **Catching too broadly**: `except Exception` or bare `except:` hides bugs. Catch the specific exception type you expect; let unexpected errors propagate.
- **Retrying non-idempotent operations**: Retry logic on a function that has side effects (database writes, API calls that create resources) can cause duplicate actions. Only retry operations that are safe to repeat.
- **Logging the error but not re-raising**: `logger.error(e)` followed by continuing execution means the caller never knows something went wrong. Log AND propagate unless you have a concrete recovery strategy.

## Checklist

- [ ] Custom exception hierarchy defined (base + 3-5 specific types)
- [ ] Boundary validation on all external input
- [ ] No bare `except:` or swallowed errors
- [ ] All loops/recursion have explicit bounds
- [ ] Exception chaining used (`from e`) to preserve tracebacks
- [ ] Errors logged once at the handler level, not at every catch
- [ ] Recovery patterns (retry/fallback) only for transient failures
- [ ] Internal functions trust their inputs (no redundant validation)
