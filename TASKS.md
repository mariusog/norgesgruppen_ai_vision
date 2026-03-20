# Tasks

**Owner**: lead-agent (exclusively). Other agents read this file but do not write to it.

Each agent has a separate plan file with detailed checklists:
- [TASKS-core.md](TASKS-core.md) -- model-agent tasks (training, weights, tuning)
- [TASKS-feature.md](TASKS-feature.md) -- inference-agent tasks (run.py, pipeline optimization)
- [TASKS-qa.md](TASKS-qa.md) -- qa-agent tasks (tests, audits, security)

## Format

Each task has: ID, status, agent, title, details, and optional dependencies.

**Statuses**: `open`, `in-progress`, `done`, `blocked`, `deferred`

## Open Tasks

| ID | Agent | Title | Details | Depends on |
|----|-------|-------|---------|------------|
| T6 | model-agent | Tune confidence and IOU thresholds | Sweep thresholds on val set, update constants.py | T1 |
| T7 | inference-agent | Benchmark and optimize for 300s budget | Profile e2e, ensure projected < 250s on L4 | T1, T2 |

## Blocked

| ID | Agent | Title | Status | Notes |
|----|-------|-------|--------|-------|
| T1 | model-agent | Monitor training jobs and download weights | blocked | GPU resources insufficient in europe-west4. Both A100 + L4 jobs stuck PENDING. Need different region. See E1 in TASKS-core.md |

## In Progress

| ID | Agent | Title | Status | Notes |
|----|-------|-------|--------|-------|
| - | - | - | - | - |

## Done

| ID | Agent | Title | Result |
|----|-------|-------|--------|
| T2 | inference-agent | Add FP16 and TensorRT export support | FP16 via HALF_PRECISION const, load_model() prefers .engine over .pt, 25/25 tests pass |
| T3 | inference-agent | Add image preprocessing module | Not needed — ultralytics handles batch loading internally, threading/multiprocessing blocked by competition |
| T4 | qa-agent | Audit run.py and write missing tests | 13 new tests: load_model, run_inference (mock model, empty results, empty boxes), image_id extraction. 33/33 pass |
| T5 | qa-agent | Security audit against extended blocklist | test_security.py expanded to full 21-module blocklist for submission files, narrower list for training. 33/33 pass |
