---
name: reproducibility
description: Use when the user asks about reproducibility, determinism, seed management, result replication, or when debugging non-deterministic behavior. Covers deterministic design, seed management, and reproducible benchmarking; examples use Python but principles are language-neutral.
---

# Reproducibility Skill

## Overview

Every result MUST be exactly reproducible. If you can't reproduce it, you can't debug it, and you can't trust it. This skill covers deterministic design, seed management, and reproducible benchmarking.

## Principle: Determinism by Default

All code should produce identical output given identical input. Non-determinism is a bug unless explicitly intended and controlled via seeds.

## Step 1: Seed Management

### Every Random Source Gets a Seed

```python
import random
import numpy as np

def run_experiment(config: dict, seed: int = 42) -> dict:
    """Run experiment with full reproducibility.

    Args:
        config: Experiment configuration.
        seed: Random seed for exact result replication.

    Returns:
        Results dict including the seed used.
    """
    # Seed ALL random sources
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Use seeded RNG throughout, never global random
    sample = rng.sample(population, k=10)
    noise = np_rng.normal(0, 1, size=(100,))

    results = do_work(config, rng=rng, np_rng=np_rng)
    results["seed"] = seed  # Always include seed in output
    return results
```

### Never Use Global Random State

```python
# BAD: Uses global random, not reproducible across runs
import random
value = random.randint(1, 100)

# GOOD: Uses seeded instance, always reproducible
rng = random.Random(seed)
value = rng.randint(1, 100)
```

### Seed Propagation

```python
def outer_function(seed: int = 42):
    rng = random.Random(seed)
    # Derive child seeds deterministically
    child_seed_1 = rng.randint(0, 2**32 - 1)
    child_seed_2 = rng.randint(0, 2**32 - 1)

    result_a = inner_function_a(seed=child_seed_1)
    result_b = inner_function_b(seed=child_seed_2)
    return combine(result_a, result_b)
```

Use the **same seed** when you want identical behavior across runs (benchmarking, regression tests). Use **derived seeds** (via `rng.randint()`) when you need independent randomness for sub-components within a single run (e.g., different entity behaviors in a simulation). Never reuse the same seed for two components that should behave independently — their random sequences would be identical.

## Step 2: Deterministic Iteration

### Dict Ordering

Python 3.7+ dicts maintain insertion order, but be careful with sets:

```python
# BAD: Set iteration order is non-deterministic
for item in some_set:
    process(item)

# GOOD: Sort for deterministic order
for item in sorted(some_set):
    process(item)

# GOOD: Use a list if order matters
items = list(some_set)
items.sort(key=lambda x: x.id)
```

### Tie-Breaking

```python
# BAD: Multiple items with same priority -- order undefined
best = min(candidates, key=lambda c: c.distance)

# GOOD: Deterministic tie-breaking with secondary key
best = min(candidates, key=lambda c: (c.distance, c.id))
```

## Step 3: Reproducible Benchmarking

### Benchmark Runner Pattern

```python
def run_benchmark(
    config: dict,
    seeds: list[int] | None = None,
    num_seeds: int = 10,
) -> dict:
    """Run benchmark with reproducible seeds.

    Args:
        config: Benchmark configuration.
        seeds: Explicit seed list. If None, generates deterministically.
        num_seeds: Number of seeds if generating.

    Returns:
        Results with per-seed and aggregate metrics.
    """
    if seeds is None:
        rng = random.Random(0)  # Deterministic seed generation
        seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_seeds)]

    results = []
    for seed in seeds:
        result = run_single(config, seed=seed)
        result["seed"] = seed
        results.append(result)

    return {
        "seeds": seeds,
        "per_seed": results,
        "mean_score": sum(r["score"] for r in results) / len(results),
        "min_score": min(r["score"] for r in results),
        "max_score": max(r["score"] for r in results),
    }
```

### Before/After Comparison

```python
def compare_runs(baseline: dict, experiment: dict) -> dict:
    """Compare two benchmark runs with the same seeds."""
    assert baseline["seeds"] == experiment["seeds"], "Seeds must match for comparison"

    diffs = []
    for b, e in zip(baseline["per_seed"], experiment["per_seed"]):
        diffs.append({
            "seed": b["seed"],
            "baseline_score": b["score"],
            "experiment_score": e["score"],
            "delta": e["score"] - b["score"],
        })

    return {
        "per_seed": diffs,
        "mean_delta": sum(d["delta"] for d in diffs) / len(diffs),
        "improved_count": sum(1 for d in diffs if d["delta"] > 0),
        "regressed_count": sum(1 for d in diffs if d["delta"] < 0),
    }
```

## Step 4: Log Everything Needed for Replay

### Run Metadata

Every run MUST log:

```python
metadata = {
    "seed": seed,
    "config": config,
    "timestamp": datetime.now().isoformat(),
    "git_commit": get_git_hash(),
    "python_version": sys.version,
    "dependencies": get_installed_packages(),
    "results_summary": summary,
}

with open(f"logs/run_{timestamp}.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### Git Hash for Code Version

```python
import subprocess

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"
```

If git is not available (e.g., CI container, exported archive), use `"unknown"` as the `git_commit` value. The key requirement is that the seed and config are always recorded — git hash is helpful but not blocking.

## Step 5: Testing Reproducibility

### Test Determinism

```python
def test_function_is_deterministic():
    """Same input always produces same output."""
    state = make_state(seed=42)
    result1 = process(state)
    result2 = process(state)
    assert result1 == result2

def test_seed_produces_same_result():
    """Same seed always produces same benchmark score."""
    score1 = run_single(config, seed=123)["score"]
    score2 = run_single(config, seed=123)["score"]
    assert score1 == score2

def test_different_seeds_produce_different_results():
    """Different seeds should generally produce different results."""
    score1 = run_single(config, seed=1)["score"]
    score2 = run_single(config, seed=2)["score"]
    # Not guaranteed but should differ in practice
    # This test validates that the seed actually affects behavior
    assert score1 != score2
```

### Regression Test Pattern

```python
@pytest.mark.slow
def test_benchmark_score_regression():
    """Benchmark scores must not regress from known baselines."""
    # Derive baselines from a known-good run: run the benchmark with your
    # chosen seeds, record the scores, and hardcode them as the baseline.
    # Update baselines only after a deliberate optimization — never
    # silently after a regression.
    BASELINE_SCORES = {
        "easy": 150,
        "medium": 100,
        "hard": 60,
    }
    SEEDS = [42, 123, 456, 789, 1024]

    for difficulty, min_score in BASELINE_SCORES.items():
        scores = [
            run_single({"difficulty": difficulty}, seed=s)["score"]
            for s in SEEDS
        ]
        mean_score = sum(scores) / len(scores)
        assert mean_score >= min_score, (
            f"{difficulty}: {mean_score:.1f} < {min_score} baseline"
        )
```

## Step 6: Environment Reproducibility

### Pin Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "numpy==1.26.4",
    "pandas==2.2.0",
]
```

### Document the Environment

```bash
# Generate exact environment snapshot
pip freeze > requirements-lock.txt
```

### Container Reproducibility

```dockerfile
FROM python:3.12-slim
COPY requirements-lock.txt .
RUN pip install -r requirements-lock.txt
```

## Gotchas

- **Reusing the same seed for independent components**: If two subsystems use the same seed, they produce identical random sequences. Use derived seeds (via `rng.randint()`) so each component gets independent randomness.
- **Assuming file system ordering is deterministic**: `os.listdir()` and `glob()` return files in arbitrary order on some platforms. Always sort the results when order matters for reproducibility.
- **Hardcoding baselines from an unvalidated run**: Baseline scores should come from a deliberate, documented run with a known-good configuration. Never copy numbers from a debug session or a run with uncommitted changes.

## Checklist

- [ ] All random sources use seeded RNG instances (never global `random`)
- [ ] Seeds propagated to all sub-functions
- [ ] Seed included in all run metadata and log files
- [ ] Set iteration sorted for deterministic order
- [ ] Tie-breaking uses secondary keys for determinism
- [ ] Benchmarks use fixed seed lists for comparability
- [ ] Before/after comparisons use the same seeds
- [ ] Run metadata includes: seed, config, git hash, timestamp
- [ ] Tests verify determinism (same input -> same output)
- [ ] Regression tests prevent score degradation
- [ ] Dependencies pinned for environment reproducibility
