---
name: debug-visualization
description: Use when the user asks to visualize data, build debug tools, inspect system behavior, or when user mentions visualization, debug tools, replay, or analysis tools. Builds text-based debug and visualization tools optimized for AI agent consumption; examples use Python but principles are language-neutral.
---

# Debug Visualization Skill

## Overview

AI agents can't open browsers, view images, or interact with GUIs. All visualization must be **text-based** and **token-budgeted**. Design every tool to be readable by an agent in a terminal.

## Core Principles

1. **Text-only**: ASCII grids, markdown tables, compact summaries
2. **Layered**: Summary first, drill-down on demand (never dump everything)
3. **Bounded**: Every command has a maximum output size (pipe through `tail`)
4. **Pre-computed**: Agents read report files, not live stdout
5. **Compressed**: Run-length encoding, aggregation, anomaly-only views

## AI-Agent Workflow (document this in CLAUDE.md)

```
Step 1: Read the report file
    cat docs/benchmark_results.md
    (5-15 lines, all key metrics and problems)

Step 2: If problems found, get the problem list
    python analyze.py <log> --problems 2>&1 | tail -20
    (compact anomaly list, 1 line per problem)

Step 3: Drill into a specific problem
    python analyze.py <log> --entity 3 --brief 2>&1 | tail -15
    (compressed timeline for one entity)

Step 4: Only if still unclear, look at a step range
    python analyze.py <log> --steps 40-50 2>&1 | tail -20
    (10 rows of detail, not 200)
```

An agent should almost NEVER need to go past Step 2. If it does, the summary report is insufficient and should be improved.

## Step 1: CLI Analysis Tool

### Architecture

```python
#!/usr/bin/env python3
"""Analyze run logs. All output is text-based and token-budgeted.

Usage:
    python analyze.py --list                     # 1-line per log
    python analyze.py <log>                      # Summary (max 30 lines)
    python analyze.py <log> --brief              # Compact summary (max 10 lines)
    python analyze.py <log> --problems           # Anomalies only (max 20 lines)
    python analyze.py <log> --grid <step>        # ASCII grid at step N
    python analyze.py <log> --entity <id>        # Entity timeline (compressed)
    python analyze.py <log> --entity <id> --brief  # Entity timeline (max 5 lines)
    python analyze.py <log> --steps <start>-<end>  # Step range detail
    python analyze.py <log> --compare <other>    # Side-by-side diff
"""
```

### --list: One Line Per Log

```python
def list_logs() -> None:
    """One line per log. Agent scans this to pick a log to analyze."""
    for json_file in sorted(LOG_DIR.glob("*.json")):
        meta = json.loads(json_file.read_text())
        seed = meta.get("seed", "?")
        score = meta.get("results_summary", {}).get("score", "?")
        steps = meta.get("total_steps", "?")
        problems = meta.get("problem_count", 0)
        # One line, all key info, easy to scan
        print(f"  {json_file.stem}  seed={seed}  score={score}  "
              f"steps={steps}  problems={problems}")

# Output:
#   run_2026-03-09_14-30  seed=42  score=150  steps=200  problems=2
#   run_2026-03-09_14-35  seed=99  score=163  steps=200  problems=0
```

### --brief: Maximum 10 Lines

```python
def print_brief(rows: list[dict], metadata: dict) -> None:
    """Ultra-compact summary for AI agent token budget.

    Target: 10 lines or fewer. Just the numbers that matter.
    """
    summary = metadata.get("results_summary", {})
    seed = metadata.get("seed", "?")

    print(f"seed={seed}  steps={len(rows)}  score={summary.get('score', '?')}")

    # Key metrics on one line each
    for key in ["idle_pct", "waste_pct", "throughput", "latency_p95"]:
        if key in summary:
            print(f"  {key}={summary[key]}")

    # Problems: one line per type (grouped)
    problems = metadata.get("problems", [])
    if problems:
        counts = {}
        for p in problems:
            counts[p["type"]] = counts.get(p["type"], 0) + 1
        problem_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  problems: {problem_str}")
    else:
        print("  problems: none")
```

### Default summary: Maximum 30 Lines

```python
def print_summary(rows: list[dict], metadata: dict) -> None:
    """Standard summary with key metrics and problem details."""
    # First print the brief version
    print_brief(rows, metadata)

    # Then add problem details (one line each, not verbose)
    problems = metadata.get("problems", [])
    if problems:
        print(f"\nProblems ({len(problems)}):")
        for p in problems:
            print(f"  [{p['type']}] step={p.get('start_step','?')} "
                  f"entity={p.get('entity_id','-')} {p.get('detail','')}")
```

### --problems: Anomaly-Only View

```python
def print_problems(rows: list[dict], metadata: dict) -> None:
    """Only anomalies. Nothing else. One line per problem."""
    problems = metadata.get("problems", [])
    if not problems:
        print("No problems detected.")
        return

    print(f"Problems: {len(problems)}")
    for p in problems:
        print(f"  [{p['type']}] step={p.get('start_step','?')} "
              f"entity={p.get('entity_id','-')} {p.get('detail','')}")
```

## Step 2: ASCII Grid Visualization

AI agents read ASCII grids well. Keep them compact.

```python
def print_grid(state: dict, width: int, height: int) -> None:
    """Print compact ASCII grid. One character per cell.

    Legend printed on one line above the grid, not as a separate block.
    Grid uses single characters with no spaces for compact display.
    """
    grid = [["." for _ in range(width)] for _ in range(height)]

    # Walls
    for wx, wy in state.get("walls", []):
        if 0 <= wx < width and 0 <= wy < height:
            grid[wy][wx] = "#"

    # Items (shelves)
    for item in state.get("items", []):
        ix, iy = item["position"]
        if 0 <= ix < width and 0 <= iy < height:
            grid[iy][ix] = "i"

    # Entities (overwrite -- they're the focus)
    for entity in state.get("entities", []):
        ex, ey = entity["position"]
        if 0 <= ex < width and 0 <= ey < height:
            grid[ey][ex] = str(entity["id"])[-1]

    # Special positions
    if "target" in state:
        tx, ty = state["target"]
        if 0 <= tx < width and 0 <= ty < height:
            grid[ty][tx] = "T"

    # Print with row numbers (compact: no spaces between cells)
    print(f"  {''.join(str(c % 10) for c in range(width))}  "
          f"[.=empty #=wall i=item 0-9=entity T=target]")
    for y in range(height):
        print(f"{y:2d}{''.join(grid[y])}")

# Output (11x9 grid, very compact):
#   01234567890  [.=empty #=wall i=item 0-9=entity T=target]
#  0...........
#  1.i.i.i.i.i.
#  2...0.......
#  3.i.i.i.i.i.
#  4.....1.....
#  5.i.i.i.i.i.
#  6...........
#  7.i.i.i.i.i.
#  8.........T.
```

Always validate grid dimensions before rendering. If width or height are not provided in the state data, infer them from the maximum coordinates in the dataset, or skip grid rendering and log a warning. Never crash on missing dimensions.

### Grid Diff (compare two steps)

```python
def print_grid_diff(state_a: dict, state_b: dict, width: int, height: int) -> None:
    """Show only cells that changed between two steps.

    Much more token-efficient than printing two full grids.
    """
    changes = []
    for entity in state_b.get("entities", []):
        eid = entity["id"]
        pos_b = tuple(entity["position"])
        # Find same entity in state_a
        for ea in state_a.get("entities", []):
            if ea["id"] == eid:
                pos_a = tuple(ea["position"])
                if pos_a != pos_b:
                    changes.append(f"  entity {eid}: ({pos_a[0]},{pos_a[1]})"
                                   f" -> ({pos_b[0]},{pos_b[1]})")
                break

    if changes:
        print(f"Changes ({len(changes)}):")
        for c in changes:
            print(c)
    else:
        print("No changes.")
```

## Step 3: Compressed Entity Timeline

Run-length encoding is critical for AI agents. A 200-step timeline compresses to 5-10 lines.

```python
def print_entity_timeline(
    rows: list[dict],
    entity_id: int,
    brief: bool = False,
) -> None:
    """Print compressed action timeline for one entity.

    Uses run-length encoding: 'move x12 | pickup | move x5 | deliver'
    Full mode: one line per action run with step ranges.
    Brief mode: single-line compressed summary.
    """
    entity_rows = [r for r in rows if int(r.get("eid", -1)) == entity_id]
    if not entity_rows:
        print(f"Entity {entity_id}: not found")
        return

    # Build runs
    runs = []
    current_action = entity_rows[0].get("act", "?")
    run_start = int(entity_rows[0].get("s", 0))
    count = 1

    for row in entity_rows[1:]:
        action = row.get("act", "?")
        if action == current_action:
            count += 1
        else:
            runs.append((current_action, run_start, count))
            current_action = action
            run_start = int(row.get("s", 0))
            count = 1
    runs.append((current_action, run_start, count))

    if brief:
        # Single line: move x12 | pickup | move x5 | deliver x3
        parts = [f"{act} x{c}" if c > 1 else act for act, _, c in runs]
        print(f"Entity {entity_id}: {' | '.join(parts)}")
    else:
        # Multi-line with step ranges
        print(f"Entity {entity_id} ({len(entity_rows)} steps):")
        for act, start, count in runs:
            end = start + count - 1
            label = f"{act} x{count}" if count > 1 else act
            print(f"  s{start:3d}-{end:3d}: {label}")

# Brief output (1 line, ~30 tokens):
#   Entity 3: move x12 | pickup | move x5 | pickup | move x8 | deliver x3 | idle x20

# Full output (~10 lines, ~80 tokens):
#   Entity 3 (50 steps):
#     s  0- 11: move x12
#     s 12- 12: pickup
#     s 13- 17: move x5
#     s 18- 18: pickup
#     s 19- 26: move x8
#     s 27- 29: deliver x3
#     s 30- 49: idle x20
```

## Step 4: Run Comparison

Show only the delta, not two full reports.

```python
def print_comparison(log_a: str, log_b: str) -> None:
    """Compare two runs. Show only metrics that changed.

    Token-efficient: skip unchanged metrics entirely.
    """
    _, meta_a = load_log(log_a)
    _, meta_b = load_log(log_b)

    print(f"A: {log_a} (seed={meta_a.get('seed')})")
    print(f"B: {log_b} (seed={meta_b.get('seed')})")

    if meta_a.get("seed") != meta_b.get("seed"):
        print("WARNING: Different seeds")

    sum_a = meta_a.get("results_summary", {})
    sum_b = meta_b.get("results_summary", {})

    changes = []
    for key in sorted(set(sum_a) | set(sum_b)):
        va = sum_a.get(key)
        vb = sum_b.get(key)
        if va == vb:
            continue  # Skip unchanged -- saves tokens
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            diff = vb - va
            sign = "+" if diff >= 0 else ""
            changes.append(f"  {key}: {va} -> {vb} ({sign}{diff})")
        else:
            changes.append(f"  {key}: {va} -> {vb}")

    if changes:
        print(f"\nChanged ({len(changes)}):")
        for c in changes:
            print(c)
    else:
        print("\nNo changes.")
```

## Step 5: Benchmark Report (Pre-Computed File)

This is what agents read first. Write it to a file, not stdout.

```python
def write_benchmark_report(
    results: dict,
    output: str = "docs/benchmark_results.md",
) -> None:
    """Write markdown benchmark report. Agents read this file directly.

    Keep it under 40 lines. Use tables for data. No prose.
    """
    lines = [
        "# Benchmark Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"Seeds: {results['seeds'][:5]}{'...' if len(results['seeds']) > 5 else ''}",
        "",
        "## Scores",
        "",
        "| Config | Mean | Min | Max | StdDev |",
        "|--------|------|-----|-----|--------|",
    ]

    for config_name, data in results.get("by_config", {}).items():
        lines.append(
            f"| {config_name} | {data['mean']:.1f} | {data['min']} "
            f"| {data['max']} | {data['std']:.1f} |"
        )

    # Diagnostics table (if present)
    diag = results.get("diagnostics")
    if diag:
        lines.extend([
            "",
            "## Diagnostics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for key, value in sorted(diag.items()):
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.2f} |")
            else:
                lines.append(f"| {key} | {value} |")

    # Problem summary
    problem_count = results.get("total_problems", 0)
    if problem_count > 0:
        lines.append(f"\nProblems: {problem_count} across all runs")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines))
```

## Step 6: Step-Range Drill-Down

For when the agent needs to look at specific steps.

```python
def print_step_range(
    rows: list[dict],
    start: int,
    end: int,
) -> None:
    """Print detail for a range of steps. Compact table format.

    Uses CSV-like output that agents parse efficiently.
    """
    filtered = [r for r in rows if start <= int(r.get("s", 0)) <= end]
    if not filtered:
        print(f"No data for steps {start}-{end}")
        return

    # Print as compact table
    cols = list(filtered[0].keys())
    print(",".join(cols))
    for row in filtered:
        print(",".join(str(row.get(c, "")) for c in cols))
```

## Anti-Patterns for AI Agents

| Anti-Pattern | Why It's Bad | Fix |
|---|---|---|
| Web-based visualizer as primary tool | Agent can't open browsers | ASCII grids + CLI tools |
| Matplotlib plots as primary output | Agent can't view images | Markdown tables + text summaries |
| Dumping full CSV to stdout | Thousands of tokens wasted | Pre-computed summary files |
| JSON lines for per-step data | 4x more tokens than CSV | Use CSV with short column names |
| Verbose column names | Extra tokens per header read | Use abbreviated names (s, eid, x, y) |
| No --brief mode | Agent can't control output size | Add --brief to every command |
| Unsorted output | Agent wastes tokens scanning | Sort by relevance (problems first) |
| Printing unchanged metrics in diffs | Wasted tokens on "no change" | Only show deltas |
| Full timeline without compression | 200 lines for 200 steps | Run-length encode to ~10 lines |

## Gotchas

- **Building tools that require a browser or GUI**: AI agents work in terminals. Every visualization must be text-based (ASCII grids, markdown tables, structured logs). If it can't render in a terminal, it's useless to an agent.
- **Generating output that exceeds token limits**: A 500-row table defeats the purpose of visualization. Always bound output (use `--brief` modes, `head`/`tail`, run-length encoding) and default to summary views.
- **Coupling visualization to runtime**: Debug tools should read from log files, not hook into live execution. This allows post-mortem analysis without reproducing the run.

## Checklist

- [ ] All visualization is text-based (ASCII grids, tables, summaries)
- [ ] Three-tier output: report file -> summary -> drill-down
- [ ] Agents read `docs/benchmark_results.md` FIRST (never raw logs)
- [ ] CLI tool has `--brief` mode (max 10-15 lines)
- [ ] Entity timelines use run-length encoding
- [ ] Grid visualization is compact (no spaces between cells, legend on one line)
- [ ] Run comparison shows only changed metrics
- [ ] Step-range drill-down outputs compact CSV
- [ ] All CLAUDE.md examples pipe through `tail` for bounded output
- [ ] No dependency on browsers, images, or GUI tools for core debugging
