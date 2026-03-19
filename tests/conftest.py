"""Shared test fixtures and configuration.

Copy this file to your tests/ directory and adapt to your project.
"""

import pytest

# ---------------------------------------------------------------------------
# Auto-reset global state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level caches and global state between tests.

    Adapt this to your project's modules that maintain global state.
    """
    # Example:
    # from src import core
    # core.clear_caches()
    yield
    # Teardown: reset again after test
    # core.clear_caches()


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_config():
    """Factory for creating test configurations.

    Returns a function that builds config dicts with sensible defaults.
    Override specific fields as needed.
    """

    def _make_config(**overrides):
        defaults = {
            "seed": 42,
            "max_steps": 100,
            "width": 10,
            "height": 10,
            "debug": False,
        }
        defaults.update(overrides)
        return defaults

    return _make_config


@pytest.fixture
def make_state(make_config):
    """Factory for creating test states.

    Returns a function that builds minimal valid state dicts.
    All positions are within default grid bounds (0 <= x < 10, 0 <= y < 10).
    """

    def _make_state(
        entities=None,
        items=None,
        config=None,
        **overrides,
    ):
        state = {
            "config": config or make_config(),
            "entities": entities
            or [
                {"id": 0, "position": [3, 3], "state": "idle"},
            ],
            "items": items or [],
            "step": 0,
        }
        state.update(overrides)
        return state

    return _make_state


# ---------------------------------------------------------------------------
# Helpers (importable by test modules)
# ---------------------------------------------------------------------------


def get_action(actions: list[dict], entity_id: int) -> dict:
    """Extract action for a specific entity from an action list."""
    for action in actions:
        if action.get("entity_id") == entity_id or action.get("id") == entity_id:
            return action
    raise ValueError(f"No action found for entity {entity_id}")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
