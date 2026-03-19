---
name: data-pipeline
description: Use when building data transforms, processing chains, ETL workflows, or when user mentions data pipeline, data processing, feature engineering, or data transformation. Creates data processing pipeline patterns; examples use Python but principles are language-neutral.
---

# Data Pipeline Patterns

> **Note**: Pipeline patterns (function chains, composable steps, independent testability) apply to any language. Code examples use Python.

## Pipeline Types

### 1. Function Chain (Simple)

```python
def process(raw_data):
    """Transform raw data through a chain of steps."""
    parsed = parse(raw_data)
    cleaned = clean(parsed)
    enriched = enrich(cleaned)
    return format_output(enriched)
```

### 2. Pipeline as List of Steps

```python
def build_pipeline(steps):
    """Compose a pipeline from a list of transform functions."""
    def run(data):
        for step in steps:
            data = step(data)
        return data
    return run

pipeline = build_pipeline([
    normalize,
    compute_features,
    rank,
    select_top,
])

result = pipeline(raw_data)
```

### 3. Generator Pipeline (Memory-Efficient)

```python
def read_records(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def filter_valid(records):
    for record in records:
        if record.get("valid"):
            yield record

def transform(records):
    for record in records:
        yield {"id": record["id"], "score": compute_score(record)}

# Compose -- nothing runs until consumed
pipeline = transform(filter_valid(read_records("data.jsonl")))
results = list(pipeline)
```

### Error Handling in Pipelines

If a pipeline step raises an exception:
- **Fail-fast** (default): Let the exception propagate. The pipeline stops and the caller sees which step failed.
- **Skip-and-log**: Wrap individual steps in try/except, log the error, and skip the failing record. Use this only for non-critical data where partial results are acceptable.
- **Never silently swallow errors** — at minimum log which record failed and why.

Choose fail-fast unless the user explicitly requests partial results.

## State Processing Pattern

For real-time systems:

```python
from dataclasses import dataclass

@dataclass
class ProcessedState:
    """Intermediate state after processing raw input."""
    entities: dict[int, tuple[int, int]]
    needed_items: dict[str, int]
    candidates: list[tuple[dict, float]]

def process_state(raw: dict) -> ProcessedState:
    entities = {e["id"]: tuple(e["position"]) for e in raw["entities"]}
    needed = compute_needed(raw)
    candidates = find_candidates(raw, needed)
    return ProcessedState(
        entities=entities,
        needed_items=needed,
        candidates=candidates,
    )
```

## ML Feature Pipeline

### Pandas-Based

```python
import pandas as pd

def prepare_dataset(path: str) -> pd.DataFrame:
    return (
        pd.read_csv(path)
        .pipe(clean_data)
        .pipe(engineer_features)
    )

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["target"]).drop_duplicates().reset_index(drop=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = (df["x"] ** 2 + df["y"] ** 2) ** 0.5
    return df
```

### NumPy-Based (Performance)

```python
import numpy as np

def vectorized_distances(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = positions - target
    return np.sqrt(np.sum(diff ** 2, axis=1))
```

## Testing Data Pipelines

### Test Each Step Independently

```python
def test_clean_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "target": [1, 1, 2]})
    result = clean_data(df)
    assert len(result) == 2

def test_compute_needed_with_empty_input():
    assert compute_needed({}) == {}
```

### Test End-to-End

```python
def test_full_pipeline():
    raw = make_sample_data()
    result = pipeline(raw)
    assert len(result) > 0
    assert all("score" in r for r in result)
```

### Test Determinism

```python
def test_pipeline_is_deterministic():
    raw = make_sample_data(seed=42)
    result1 = pipeline(raw)
    result2 = pipeline(raw)
    assert result1 == result2
```

## Gotchas

- **Mutating input data in pipeline steps**: A step that modifies its input in place breaks replayability and makes debugging impossible. Each step should return new data, not modify the input.
- **Building the full pipeline before testing any step**: Test each transformation step in isolation first. Debugging a 10-step pipeline failure is much harder than debugging one step.
- **Silently dropping records**: A step that filters without logging how many records were dropped hides data loss. Always log the count before and after filtering.

## Checklist

- [ ] Each pipeline step is a pure function (input -> output)
- [ ] Steps are testable independently
- [ ] Intermediate data structures are well-defined (dataclasses/TypedDict)
- [ ] Large datasets use generators, not lists
- [ ] NumPy/pandas vectorization used where applicable
- [ ] Error handling at data boundaries (file I/O, API responses)
- [ ] Pipeline is deterministic for deterministic inputs
- [ ] Seeds propagated for any randomized steps
