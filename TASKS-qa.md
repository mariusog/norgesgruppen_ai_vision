# Agent Plan: qa-agent

**Owner**: qa-agent (exclusively). Lead-agent creates tasks here; you fill out checklists and results.

## Active Tasks

(none)

---

## Escalations

| Tag | Task | Description |
|-----|------|-------------|
| - | - | - |

## Completed Tasks

### T4: Audit run.py and write missing tests
**Status**: done
**Branch**: `qa/T4-audit-run-py`
**Target**: All public functions in run.py have unit tests with edge cases

- [x] Read `run.py` -- public functions: `load_model`, `collect_images`, `run_inference`, `main`
- [x] Read `tests/test_run.py` -- existing coverage: only `collect_images` (5 tests)
- [x] Identify gaps and write tests:
  - [x] `load_model`: `test_load_model_file_not_found` -- FileNotFoundError when weights missing
  - [x] `run_inference`: `test_run_inference_with_mock_model` -- mock YOLO returning known boxes, verify output dict format (keys, types, bbox xywh conversion)
  - [x] `run_inference`: `test_run_inference_empty_results` -- model returns None boxes -> empty list
  - [x] `run_inference`: `test_run_inference_empty_boxes` -- model returns empty boxes list -> empty list
  - [x] Image ID extraction: `test_image_id_from_img_00042` (->42), `test_image_id_from_img_00001` (->1), `test_image_id_from_img_00100` (->100), `test_image_id_from_img_00000` (->0)
- [x] All 33 tests pass
- [x] Audit code quality:
  - No SOLID violations -- functions are small and single-purpose
  - No magic numbers -- all thresholds from constants.py
  - Type annotations on all public functions
  - File size: 113 lines (well under 300)
  - All functions under 30 lines
  - Note: `load_model` has TensorRT engine fallback logic (checks MODEL_ENGINE_PATH first, then MODEL_PATH)

**Result**: 8 new tests added covering load_model, run_inference (format, empty results, empty boxes), and image_id extraction (4 filename variants). All 33 tests pass.

---

### T5: Security audit against extended blocklist
**Status**: done
**Branch**: `qa/T5-security-audit`
**Target**: Zero violations against full competition blocklist

- [x] Read extended blocklist from competition docs (MCP `search_docs` returned no results; used blocklist from session_handoff.md and TASKS-qa.md)
  - Blocked modules (submission): `os`, `subprocess`, `socket`, `ctypes`, `builtins`, `sys`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`
  - Blocked builtins: `eval()`, `exec()`, `compile()`, `__import__()`
- [x] Updated `tests/test_security.py`:
  - Split into two parametrized tests: `test_submission_no_forbidden_imports` (extended blocklist, scans run.py + src/*.py) and `test_training_no_forbidden_imports` (narrow blocklist, scans training/*.py)
  - Submission files scanned against all 22 blocked modules
  - Training files scanned against original 5 blocked modules (they legitimately use shutil, etc.)
- [x] Verified all submission files (run.py, src/__init__.py, src/constants.py) pass extended blocklist
- [x] All 33 tests pass
- [x] Dependencies note: ultralytics internally uses os, sys, etc. at import time, but the security scanner only checks our source files, not third-party packages. This is by design -- ultralytics is pre-installed in the sandbox.

**Result**: test_security.py updated with full 22-module blocklist for submission files, narrower 5-module blocklist for training files. All tests pass. No violations found.
