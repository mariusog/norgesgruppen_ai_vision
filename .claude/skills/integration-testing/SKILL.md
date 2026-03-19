---
name: integration-testing
description: Use when testing API integrations, service interactions, database access, or when user mentions integration test, mock API, test containers, contract test, or end-to-end test. Covers module interactions, external services, and end-to-end workflows; examples use Python/pytest but principles are language-neutral.
---

# Integration Testing Patterns

> **Note**: Integration testing principles (isolate external dependencies, test contracts, use realistic fixtures) apply to any language. Code examples use Python/pytest.

## When to Use Integration Tests

| Situation | Test Type |
|-----------|-----------|
| Pure function, single input/output | Unit test |
| Two modules working together | Integration test |
| External API or service call | Integration test (mocked) |
| Full request-to-response flow | End-to-end test |
| Data pipeline with multiple stages | Integration test |

## Mocking External Services

### Basic Mock with pytest

```python
from unittest.mock import patch, MagicMock

def test_fetch_data_handles_timeout():
    """Test behavior when external API times out."""
    with patch("mymodule.requests.get") as mock_get:
        mock_get.side_effect = requests.Timeout("Connection timed out")
        result = fetch_data("https://api.example.com/data")
        assert result is None  # Graceful degradation

def test_fetch_data_returns_parsed_response():
    """Test normal API response is parsed correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": [{"id": 1, "name": "test"}]}
    with patch("mymodule.requests.get", return_value=mock_response):
        result = fetch_data("https://api.example.com/data")
        assert len(result) == 1
        assert result[0]["id"] == 1
```

Mock at the **caller's boundary**, not at the network layer. If your code calls `api_client.get_user()`, mock `api_client.get_user`, not `requests.get`. This keeps tests focused on your code's behavior, not on the HTTP library's internals. Only mock at the network level (`socket`, `responses`) when testing the HTTP client itself.

### Fixture-Based Mock Service

```python
import pytest

@pytest.fixture
def mock_api():
    """Reusable mock API that returns configurable responses."""
    responses = {}

    class MockAPI:
        def get(self, endpoint):
            if endpoint in responses:
                return responses[endpoint]
            raise ConnectionError(f"No mock for {endpoint}")

        def set_response(self, endpoint, data):
            responses[endpoint] = data

    return MockAPI()

def test_pipeline_with_api(mock_api):
    mock_api.set_response("/items", [{"id": 1, "score": 0.9}])
    result = process_pipeline(api=mock_api)
    assert result.top_item_id == 1
```

## File/IO Testing

### Temporary Files

```python
import tempfile
from pathlib import Path

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test data files."""
    data_file = tmp_path / "input.csv"
    data_file.write_text("id,score\n1,0.9\n2,0.7\n3,0.5\n")
    config_file = tmp_path / "config.json"
    config_file.write_text('{"seed": 42, "max_steps": 100}')
    return tmp_path

def test_load_and_process(temp_data_dir):
    result = load_and_process(temp_data_dir / "input.csv")
    assert len(result) == 3
    assert result[0]["id"] == 1
```

### Testing File Output

```python
def test_export_writes_valid_csv(tmp_path):
    output_path = tmp_path / "output.csv"
    data = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.7}]
    export_csv(data, output_path)

    lines = output_path.read_text().strip().split("\n")
    assert lines[0] == "id,score"  # Header
    assert len(lines) == 3  # Header + 2 rows
```

## Database Testing

### In-Memory Database

```python
import sqlite3

@pytest.fixture
def test_db():
    """In-memory SQLite database with schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    conn.execute("INSERT INTO items VALUES (1, 'widget', 0.9)")
    conn.execute("INSERT INTO items VALUES (2, 'gadget', 0.7)")
    conn.commit()
    yield conn
    conn.close()

def test_query_top_items(test_db):
    result = query_top_items(test_db, min_score=0.8)
    assert len(result) == 1
    assert result[0]["name"] == "widget"
```

### Transactional Isolation

```python
@pytest.fixture
def db_session(test_db):
    """Each test runs in a transaction that rolls back after."""
    test_db.execute("BEGIN")
    yield test_db
    test_db.execute("ROLLBACK")
```

## Contract Testing

Verify that your code handles the real API's response shape correctly.

```python
# Save a real API response once, use it as a fixture forever
REAL_API_RESPONSE = {
    "status": "ok",
    "data": {
        "items": [
            {"id": "abc", "type": "widget", "position": [1, 2], "properties": {}},
        ],
        "metadata": {"version": "2.1", "timestamp": 1234567890},
    },
}

def test_parser_handles_real_response_shape():
    """Ensure our parser handles the actual API response format."""
    result = parse_api_response(REAL_API_RESPONSE)
    assert len(result.items) == 1
    assert result.items[0].id == "abc"
    assert result.items[0].position == (1, 2)

def test_parser_handles_missing_optional_fields():
    """API sometimes omits optional fields."""
    response = {
        "status": "ok",
        "data": {
            "items": [{"id": "abc", "type": "widget", "position": [1, 2]}],
            # metadata missing -- should still parse
        },
    }
    result = parse_api_response(response)
    assert result.metadata is None
```

Refresh contract fixtures when: (1) the external API releases a new version you're upgrading to, (2) a contract test fails unexpectedly (the API may have changed), or (3) at a regular cadence (e.g., quarterly). Store the fixture alongside the test with a comment noting when it was captured.

## Pipeline Integration Tests

Test multiple stages working together:

```python
def test_full_pipeline_produces_valid_output():
    """Test the complete data pipeline end-to-end."""
    raw = make_sample_input(seed=42)

    # Run the full pipeline
    result = pipeline(raw)

    # Verify output structure
    assert len(result) > 0
    assert all("id" in r and "score" in r for r in result)
    # Verify ordering
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)

def test_pipeline_stages_compose_correctly():
    """Test that stage outputs match next stage's expected input."""
    raw = make_sample_input(seed=42)

    parsed = parse_stage(raw)
    assert isinstance(parsed, list)
    assert all(isinstance(r, ParsedRecord) for r in parsed)

    cleaned = clean_stage(parsed)
    assert len(cleaned) <= len(parsed)

    scored = score_stage(cleaned)
    assert all(hasattr(r, "score") for r in scored)
```

## Test Environment Isolation

### Environment Variables

```python
@pytest.fixture
def clean_env(monkeypatch):
    """Ensure tests don't leak environment state."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ENV", "test")

def test_config_uses_defaults_without_env(clean_env):
    config = load_config()
    assert config.database_url == "sqlite://:memory:"
```

### Network Isolation

```python
@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Prevent accidental real network calls in tests."""
    def blocked(*args, **kwargs):
        raise RuntimeError("Tests must not make real network calls")
    monkeypatch.setattr("socket.socket.connect", blocked)
```

## Slow Test Management

```python
import pytest

# Mark integration tests that are slow
@pytest.mark.slow
def test_full_pipeline_with_large_dataset():
    data = make_large_dataset(n=10000, seed=42)
    result = pipeline(data)
    assert len(result) > 0

# Run fast tests only (default):  pytest -m "not slow"
# Run all tests:                  pytest
# Run only slow tests:            pytest -m slow
```

Mark a test as `slow` if it takes more than 2 seconds to run. The exact threshold is project-specific — adjust based on your test suite's total runtime budget. The goal: the fast test suite (`-m 'not slow'`) completes in under 30 seconds.

## Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Real network calls in tests | Mock all HTTP/socket calls |
| Shared mutable test database | Use transactional rollback or fresh DB per test |
| Tests depending on execution order | Each test creates its own state |
| Testing mock behavior instead of real behavior | Verify your code's logic, not the mock |
| Huge integration tests that test everything | Split into focused tests per integration point |
| No contract tests for external APIs | Save real response samples, test parser against them |
| Skipping cleanup (temp files, connections) | Use fixtures with teardown (`yield` + cleanup) |

## Gotchas

- **Mocking at the wrong level**: Mocking deep internals (e.g., `socket.connect`) instead of the caller's boundary (e.g., `api_client.get_user`) makes tests brittle and tightly coupled to implementation.
- **Test data leaking between tests**: Database state from one test affecting another causes intermittent failures. Each test must set up and tear down its own state — use transactions or fixtures with cleanup.
- **Tests that depend on execution order**: If test B only passes after test A, you have a hidden dependency. Every test must be independently runnable with `pytest test_file.py::test_name`.

## Checklist

- [ ] External services mocked (no real network calls in CI)
- [ ] File I/O uses `tmp_path` / temporary directories
- [ ] Database tests use in-memory DB or transactional rollback
- [ ] Contract tests exist for each external API response shape
- [ ] Pipeline tests verify stage composition (output-to-input compatibility)
- [ ] Environment variables isolated with `monkeypatch`
- [ ] Slow tests marked and excludable (`@pytest.mark.slow`)
- [ ] Network calls blocked by default in test suite
- [ ] Each test is independent (no shared mutable state)
