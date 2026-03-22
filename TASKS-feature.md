# Agent Plan: inference-agent

**Owner**: inference-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

### T2: Add FP16 and TensorRT export support
**Status**: done
**Branch**: `Initialize-claude-and-mcp`
**Target**: FP16 inference working; TensorRT export path ready for L4

- [x] Add `half=True` to `model.predict()` call in `run.py` for FP16 inference
- [x] Add `HALF_PRECISION = True` constant to `src/constants.py` (tag: `NEEDS CONSTANT`)
- [x] Verify FP16 doesn't break output format (scores still Python float, etc.) — output uses `.tolist()` and `float()`, always returns Python float
- [x] Add TensorRT export utility function (for use when weights are available):
  - `model.export(format="engine", imgsz=IMAGE_SIZE, half=True, device=0)`
  - Output to `weights/model.engine`
  - Add `MODEL_ENGINE_PATH` constant
- [x] Update `load_model()` to prefer `.engine` file if it exists, fallback to `.pt`
- [x] Self-review: lint + quality check — `ruff check` passes
- [x] Tests pass — 25/25
- [x] No forbidden imports introduced

**Result**: FP16 enabled via `HALF_PRECISION` constant. `load_model()` checks for TensorRT `.engine` first, falls back to `.pt`. TensorRT export can be done with `model.export(format="engine", imgsz=IMAGE_SIZE, half=True, device=0)` once weights are available.

---

### T3: Add image preprocessing module
**Status**: not-needed
**Branch**: n/a
**Target**: Reduce I/O bottleneck with threaded image pre-fetch

- [x] Consider: is threaded pre-fetch worth it given ultralytics handles loading internally?
  - **Finding**: ultralytics `model.predict()` already handles batch image loading, resizing, and preprocessing internally when passed a list of file paths. Creating a custom `src/dataset.py` would duplicate this work with no benefit.
  - **Blocker**: `threading` and `multiprocessing` are on the competition blocklist, so any custom threaded pre-fetch is impossible in submission code.
  - **Decision**: Task is not needed. Image path collection is already handled by `collect_images()` in `run.py`. No `src/dataset.py` file created.

**Result**: Not needed. Ultralytics handles batch image loading internally. Threaded pre-fetch is blocked by competition security restrictions (`threading`/`multiprocessing` forbidden). No action required.

---

### T7: Benchmark and optimize for 300s budget
**Status**: blocked (depends on T1 weights + T2 FP16)
**Branch**: `feature/T7-benchmark-optimize`
**Target**: Projected total inference < 250s on L4

- [ ] Run `/benchmark-inference` with model weights
- [ ] Profile: is bottleneck model forward pass, image loading, or postprocessing?
- [ ] If model forward pass: try TensorRT engine, adjust batch size
- [ ] If I/O: optimize image loading path
- [ ] Log p50/p95/total to `docs/benchmark_results.md`
- [ ] Run `/validate-submission` — must pass all checks
- [ ] Tests pass

**Result**: pending

---

## Escalations

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks

(none yet)
