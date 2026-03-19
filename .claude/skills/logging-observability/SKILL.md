---
name: logging-observability
description: Use when adding logging, debugging output, metrics collection, or when user mentions logging, observability, diagnostics, monitoring, or debug data. Implements structured logging, diagnostics, and observability patterns optimized for AI agent consumption; examples use Python but principles are language-neutral.
---

# Logging and Observability Skill

## Overview

Every system needs enough logging to answer "what happened and why?" after the fact.
Logs serve two audiences: **humans** (interactive debugging) and **AI agents** (automated analysis with token budgets). Design for both.

## Core Principles

1. **Never use bare `print()`** -- use `logging` module
2. **CSV for tabular data, JSON for metadata** -- never JSON lines for high-volume data
3. **Summarization layers** -- agents read pre-computed summaries, never raw logs
4. **Write reports to files** -- agents read files directly, not stdout
5. **Bounded output** -- every CLI tool has a `--brief` mode for agents

## Token-Efficient Log Formats

### Format Comparison

```
100 rows of entity data (id, x, y, action, score):

JSON lines:  ~8,500 tokens  (keys repeated every row)
CSV:         ~2,100 tokens  (headers once, commas between values)
TSV:         ~2,000 tokens  (headers once, tabs between values)

CSV is 4x more token-efficient than JSON lines for tabular data.
```

### Rule: Use the Right Format for the Right Data

| Data Type | Format | Why |
|-----------|--------|-----|
| Per-step tabular data | **CSV** | Headers once, minimal overhead per row |
| Run metadata (config, seed, summary) | **JSON** | One-off, structured, small |
| Pre-computed summaries | **Markdown** | Agents read directly, human-readable |
| Anomaly lists | **CSV or markdown table** | Compact, scannable |

### NEVER Use JSON Lines for High-Volume Data

```python
# BAD: ~85 tokens per row (keys repeated)
{"step": 1, "entity_id": 0, "x": 3, "y": 5, "action": "move_right", "score": 10}
{"step": 2, "entity_id": 0, "x": 4, "y": 5, "action": "pickup", "score": 11}

# GOOD: ~21 tokens per row (headers once)
step,entity_id,x,y,action,score
1,0,3,5,move_right,10
2,0,4,5,pickup,11
```

## Step 1: Three-Tier Log Architecture

Every run produces three outputs at different granularity:

### Tier 1: Summary Report (agents read this FIRST)

Pre-computed markdown in `docs/`. Agent reads this file -- never parses raw logs.

```python
def write_summary_report(metadata: dict, diagnostics: dict, path: str) -> None:
    """Write pre-computed summary report as markdown.

    This is the PRIMARY output for AI agents. It must contain enough
    information to decide whether to drill down into raw logs.
    """
    lines = [
        f"# Run Summary",
        f"",
        f"seed={metadata['seed']}  steps={metadata['total_steps']}  "
        f"score={metadata['score']}",
        f"",
        f"## Key Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for key, value in sorted(diagnostics.items()):
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.2f} |")
        else:
            lines.append(f"| {key} | {value} |")

    # Problems section -- only if problems exist
    problems = metadata.get("problems", [])
    if problems:
        lines.extend([
            f"",
            f"## Problems ({len(problems)})",
            f"",
            f"| Type | Step | Entity | Details |",
            f"|------|------|--------|---------|",
        ])
        for p in problems:
            lines.append(
                f"| {p['type']} | {p.get('start_step', '?')} "
                f"| {p.get('entity_id', '-')} | {p.get('detail', '')} |"
            )
    else:
        lines.append(f"\nNo problems detected.")

    Path(path).write_text("\n".join(lines))
```

### Tier 2: CSV Detail Log (agents drill into this on demand)

Per-step data for when the summary flags a problem. Use short column names.

```python
import csv

def write_csv_log(path: str, rows: list[dict]) -> None:
    """Write per-step data as CSV.

    Column names should be short to save tokens:
    - 's' not 'step_number', 'eid' not 'entity_id'
    - 'x','y' not 'position_x','position_y'
    - 'act' not 'action_taken'
    """
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
```

**Short column names save tokens:**
```
# BAD: Long column names (~30 tokens for header)
step_number,entity_id,position_x,position_y,action_taken,current_score

# GOOD: Short column names (~12 tokens for header)
s,eid,x,y,act,score
```

### Tier 3: JSON Metadata (config, seed, environment)

One-off data. Small, so JSON is fine.

```python
import json

def write_metadata(path: str, metadata: dict) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
```

### File Naming Convention

```
logs/
  run_2026-03-09_14-30-00.csv      # Tier 2: per-step CSV
  run_2026-03-09_14-30-00.json     # Tier 3: metadata
docs/
  benchmark_results.md              # Tier 1: summary report
```

## Step 2: AI-Agent Consumption Pattern

### How agents should read logs (document this in CLAUDE.md):

```markdown
## Analyzing Runs (for AI agents)

1. **Read the summary FIRST**: `cat docs/benchmark_results.md`
   - This has scores, key metrics, and auto-detected problems
   - If no problems, STOP -- no need to read raw logs

2. **Drill into problems**: Only if the summary flags issues
   - `python analyze.py <log> --problems`  (shows only anomalies)
   - `python analyze.py <log> --entity 3`  (timeline for problem entity)
   - `python analyze.py <log> --steps 40-60`  (drill into problem window)

3. **NEVER read raw CSV files directly** -- use the analysis tool
   - Raw CSV can be 10,000+ lines = thousands of tokens wasted
   - The analysis tool summarizes and compresses
```

### Token Budget Rules for CLI Tools

```python
# Every CLI analysis command MUST have bounded output.
# Default mode prints at most 40 lines.
# --brief mode prints at most 15 lines.
# Always pipe through tail as a safety net.

# In CLAUDE.md, document like this:
# python analyze.py <log> --brief 2>&1 | tail -15
# python analyze.py <log> --problems 2>&1 | tail -30
```

## Step 3: Diagnostic Data Collection

### DiagnosticTracker with Built-in Summarization

```python
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class DiagnosticTracker:
    """Collects per-step metrics with built-in summarization.

    The summary() method produces a token-efficient dict that agents
    can read directly without processing raw data.
    """
    step_count: int = 0
    metrics: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list),
    )
    anomalies: list[dict] = field(default_factory=list)
    enabled: bool = False

    def record(self, metric: str, value: float) -> None:
        if not self.enabled:
            return
        self.metrics[metric].append(value)

    def record_anomaly(self, kind: str, detail: str, **kwargs) -> None:
        if not self.enabled:
            return
        self.anomalies.append({
            "step": self.step_count,
            "type": kind,
            "detail": detail,
            **kwargs,
        })

    def step(self) -> None:
        self.step_count += 1

    def summary(self) -> dict:
        """Return compact aggregate statistics.

        This is what agents read. One dict, not thousands of rows.
        """
        result = {
            "total_steps": self.step_count,
            "anomaly_count": len(self.anomalies),
        }
        for name, values in self.metrics.items():
            n = len(values)
            if n == 0:
                continue
            s = sum(values)
            result[f"{name}_mean"] = round(s / n, 2)
            result[f"{name}_max"] = round(max(values), 2)
            result[f"{name}_p95"] = round(
                sorted(values)[int(n * 0.95)], 2,
            ) if n >= 20 else round(max(values), 2)
        return result

    def problems_brief(self) -> str:
        """Return compact problem summary for agent consumption.

        Groups anomalies by type and returns counts, not full details.
        """
        if not self.anomalies:
            return "No problems detected."
        counts = defaultdict(int)
        for a in self.anomalies:
            counts[a["type"]] += 1
        lines = [f"{kind}: {count}" for kind, count in sorted(counts.items())]
        return " | ".join(lines)
```

## Step 4: Run-Length Encoding for Timelines

AI agents waste tokens reading repetitive action sequences. Compress them.

```python
def compress_timeline(actions: list[str]) -> str:
    """Run-length encode an action sequence for token efficiency.

    Instead of: move,move,move,move,pickup,move,move,deliver
    Produce:    move x4 | pickup | move x2 | deliver

    This is 3x more token-efficient for typical sequences.
    """
    if not actions:
        return "(empty)"
    parts = []
    current = actions[0]
    count = 1
    for action in actions[1:]:
        if action == current:
            count += 1
        else:
            parts.append(f"{current} x{count}" if count > 1 else current)
            current = action
            count = 1
    parts.append(f"{current} x{count}" if count > 1 else current)
    return " | ".join(parts)


# Example output (fits in one line, easy for agent to scan):
# move x12 | pickup | move x5 | pickup | move x8 | deliver x3 | idle x20
```

## Step 5: Anomaly Detection

Only surface problems -- don't dump everything.

```python
IDLE_THRESHOLD = 10       # steps idle before flagging
OSCILLATION_THRESHOLD = 5  # flip-flops before flagging
SCORE_GAP_THRESHOLD = 20   # steps without progress before flagging

def detect_problems(rows: list[dict]) -> list[dict]:
    """Scan log data for anomalies. Returns compact problem list.

    Each problem dict has: type, start_step, entity_id, detail (one-liner).
    The 'detail' field is a short string, not a data dump.
    """
    problems = []

    # Idle detection
    idle_runs = find_consecutive_runs(rows, lambda r: r["act"] == "idle")
    for run in idle_runs:
        if run["length"] >= IDLE_THRESHOLD:
            problems.append({
                "type": "idle",
                "start_step": run["start"],
                "entity_id": run["entity_id"],
                "detail": f"idle x{run['length']} at ({run['x']},{run['y']})",
            })

    # Oscillation detection
    for eid, osc in find_oscillations(rows).items():
        if osc["count"] >= OSCILLATION_THRESHOLD:
            problems.append({
                "type": "oscillation",
                "start_step": osc["start"],
                "entity_id": eid,
                "detail": f"flip-flop x{osc['count']} between {osc['states']}",
            })

    return problems
```

## Step 6: Logging Setup

```python
import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # File: compact format (no JSON -- just timestamp + level + message)
    fh = logging.FileHandler(LOG_DIR / f"{name}_{timestamp}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    # Console: same compact format
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)

    return logger
```

## What to Log at Each Level

### INFO (always -- compact summaries)
- Run start: `seed=42 config={"difficulty":"hard"} steps=200`
- Run end: `score=150 steps=200 problems=2 duration=1.3s`
- Key transitions: `order_completed id=3 score_delta=+6`

### DEBUG (diagnostic mode -- per-step details go to CSV, not log)
- Expensive computations: `cache_hits=45 cache_misses=3 hit_rate=0.94`
- Decision details go to CSV, not logging stream

### WARNING (anomalies)
- `idle_run entity=3 steps=15 at=(5,2)`
- `oscillation entity=1 count=8 between=move_left,move_right`
- `timeout step=42 elapsed=2.3s budget=2.0s`

## Gotchas

- **Logging sensitive data**: Passwords, tokens, PII, and credit card numbers must never appear in logs. Scrub or mask sensitive fields before logging. This is a security violation, not just a best-practice issue.
- **String formatting instead of structured logging**: `logger.info(f"User {user_id} logged in")` loses structure. Use `logger.info("user_login", extra={"user_id": user_id})` so log aggregators can filter and search by field.
- **Logging too much in hot paths**: A log statement inside a tight loop can generate millions of entries and fill disks. Use debug-level for high-frequency events and sample or rate-limit if needed.

## Checklist

- [ ] `logging` module used everywhere (no bare `print()`)
- [ ] CSV for per-step tabular data (short column names)
- [ ] JSON only for one-off metadata (seed, config, environment)
- [ ] Pre-computed markdown summary in `docs/` (agents read this first)
- [ ] CLI analysis tool with `--brief` mode (bounded output)
- [ ] Run-length encoding for action timelines
- [ ] Anomaly detection surfaces problems as compact one-liners
- [ ] All analysis commands piped through `tail` in CLAUDE.md examples
- [ ] Agents NEVER read raw CSV -- they read summaries and drill down
- [ ] Short column names in CSV (s, eid, x, y, act -- not step_number, entity_id)
- [ ] Seed logged in every run for reproducibility
