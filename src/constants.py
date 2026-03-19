"""Named constants and tuning parameters.

All numeric thresholds, limits, and configuration values live here.
No magic numbers in logic code -- reference these constants instead.

Naming convention: UPPER_SNAKE_CASE
Sections: group related constants under comment headers.

When adding a new constant:
1. Choose a descriptive name that explains WHAT it controls
2. Add a comment explaining WHY this value was chosen
3. If the value was tuned empirically, note the benchmark that validated it
"""

# ---------------------------------------------------------------------------
# System limits
# ---------------------------------------------------------------------------

# Maximum iterations for search/exploration functions (prevents unbounded loops)
MAX_SEARCH_STEPS = 10_000

# Maximum file/data size to process in one pass
MAX_BATCH_SIZE = 1_000

# ---------------------------------------------------------------------------
# Performance tuning
# ---------------------------------------------------------------------------

# Cache size limit for LRU caches (0 = unbounded, use with caution)
DEFAULT_CACHE_SIZE = 1024

# Timeout budget for real-time operations (seconds)
OPERATION_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------

# Example: thresholds, weights, and scoring parameters go here.
# Replace with your project-specific constants.
#
# SCORE_MULTIPLIER = 1.15
# BASE_BONUS = 5
# RETRY_LIMIT = 3

# ---------------------------------------------------------------------------
# Logging and diagnostics
# ---------------------------------------------------------------------------

# Example diagnostic thresholds. Replace with your project's needs.
#
# IDLE_THRESHOLD = 10          # steps idle before flagging as anomaly
# OSCILLATION_THRESHOLD = 5   # state flip-flops before flagging
# SCORE_GAP_THRESHOLD = 20    # steps without progress before flagging
