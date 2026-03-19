---
name: caching-strategies
description: Use when adding memoization, precomputation, result caching, or when user mentions caching, performance, cache keys, or memoization. Implements caching patterns for performance optimization; examples use Python but principles are language-neutral.
---

# Caching Strategies

> **Note**: Caching patterns (memoization, precomputation, cache invalidation) apply to any language. Code examples use Python.

## Overview

Focus on these caching layers:
- **Precomputation**: Compute expensive results once at startup
- **Memoization**: Cache function results by arguments
- **Module-level caching**: Global dicts for reusable lookups
- **Disk caching**: Persist results across runs (joblib, shelve)

## Memoization

### functools.lru_cache

Best for pure functions with hashable arguments:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Check cache stats
print(fibonacci.cache_info())

# Clear if needed
fibonacci.cache_clear()
```

### functools.cache (Python 3.9+)

Shorthand for `lru_cache(maxsize=None)`:

```python
from functools import cache

@cache
def expensive_pure_function(x, y):
    return complex_calculation(x, y)
```

### When lru_cache Won't Work

For unhashable arguments (lists, dicts, sets), convert to hashable:

```python
@lru_cache(maxsize=256)
def find_path(start, goal, blocked_frozenset):
    return search(start, goal, blocked_frozenset)

# Call with frozenset
path = find_path((0, 0), (5, 5), frozenset(blocked_cells))
```

## Module-Level Dict Caching

For caches that need manual management:

```python
_dist_cache: dict[tuple, dict] = {}

def get_distances_from(source, blocked):
    if source not in _dist_cache:
        _dist_cache[source] = compute_all_distances(source, blocked)
    return _dist_cache[source]

def clear_caches():
    """Reset caches between runs/sessions."""
    _dist_cache.clear()
```

## Precomputation Pattern

Compute once at initialization, use many times:

```python
_static_data = None

def init_static(config):
    """Compute static data once -- config never changes within a session."""
    global _static_data
    _static_data = expensive_setup(config)

def get_static():
    assert _static_data is not None, "Call init_static() first"
    return _static_data
```

## Disk Caching

### joblib (for ML/data science)

```python
from joblib import Memory

memory = Memory("./cache", verbose=0)

@memory.cache
def train_model(X, y, params):
    """Cached across runs -- recomputes only if inputs change."""
    model = SomeModel(**params)
    model.fit(X, y)
    return model
```

### shelve (simple key-value persistence)

```python
import shelve

def load_or_compute(key, compute_fn):
    with shelve.open("cache.db") as db:
        if key not in db:
            db[key] = compute_fn()
        return db[key]
```

## Cache Invalidation

### Choosing an Invalidation Strategy

- Data changes are **event-driven** (e.g., user action, write operation) → **Event-based** invalidation (clear cache on write)
- Data changes are **time-based** or **unpredictable** (e.g., external API, sensor data) → **TTL-based** invalidation (cache expires after fixed duration)
- Data is **immutable** once computed (e.g., historical aggregations, hashed lookups) → **No invalidation needed** — cache permanently
- Data changes are **rare but important** (e.g., config, feature flags) → **Event-based** with a TTL safety net (clear on change, but also expire after a long TTL as a fallback)

### Event-Based

```python
def on_new_session():
    """Clear all caches when starting a new session."""
    _dist_cache.clear()
    fibonacci.cache_clear()
```

### Time-Based

```python
import time

_cache = {}
_cache_time = {}
CACHE_TTL = 300  # 5 minutes

def get_cached(key, compute_fn):
    now = time.time()
    if key in _cache and (now - _cache_time[key]) < CACHE_TTL:
        return _cache[key]
    result = compute_fn()
    _cache[key] = result
    _cache_time[key] = now
    return result
```

## Testing Caching

```python
def test_cache_returns_same_object():
    result1 = get_distances_from((0, 0), blocked)
    result2 = get_distances_from((0, 0), blocked)
    assert result1 is result2  # Same object (cached)

def test_cache_invalidation():
    get_distances_from((0, 0), blocked)
    clear_caches()
    assert (0, 0) not in _dist_cache
```

## Gotchas

- **Caching mutable objects**: If you cache a list or dict and return a reference, the caller can modify it and corrupt the cache. Return copies, or cache immutable types (tuples, frozensets).
- **Forgetting to invalidate**: Adding a cache without an invalidation path means stale data after writes. Every cache must have a clear answer to "when does this entry become invalid?"
- **Unhashable cache keys**: Passing a list or dict as a cache key will crash at runtime. Convert to tuple or frozenset before using as a key.

## Checklist

- [ ] Pure functions use `@lru_cache` or `@cache`
- [ ] Static data precomputed once at initialization
- [ ] Cache keys are hashable (tuples, frozensets, not lists/dicts)
- [ ] Caches cleared between runs/sessions
- [ ] Cache hit rates monitored for key caches (log them)
- [ ] No unbounded caches — every cache must have an explicit size limit (e.g., `maxsize=1024`) or TTL. If you can't predict the cardinality of cache keys, use an LRU policy. As a rule of thumb, a cache entry should not exceed 1KB, and total cache memory should stay under 100MB unless profiling justifies more.
- [ ] Disk caching for expensive ML training or data processing
