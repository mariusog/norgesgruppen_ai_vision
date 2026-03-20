# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Context

**NM i AI 2026 – NorgesGruppen Object Detection** (deadline March 22, 2026).
Task: detect and classify grocery products from **356 categories** (IDs 0–355; ID 355 = `unknown_product`) in shelf images.
Scoring: **70% detection mAP** (bbox IoU ≥ 0.5, category ignored) + **30% classification mAP** (IoU ≥ 0.5 AND correct category_id). Detection-only baseline scores up to 0.70.
Submission limits: **3 per day**, 2 in-flight at once.

## Commands

```bash
# Run tests (fast)
python -m pytest tests/ -q --tb=line -m "not slow" 2>&1 | tail -20

# Run a single test
python -m pytest tests/path/file.py::test_name -q --tb=short 2>&1 | tail -40

# Lint / format / type check
ruff check src/ run.py training/
ruff format src/ run.py training/
mypy src/ run.py

# Run inference locally (requires weights/model.pt)
python run.py --input /path/to/images/ --output /tmp/predictions.json

# Pre-submission checklist
bash scripts/validate_submission.sh

# Download dataset from GCS
bash scripts/download_dataset.sh

# Build submission zip
python scripts/create_submission.py

# Pre-compute prototype embeddings from reference images
python scripts/precompute_prototypes.py

# Launch Vertex AI training job (see model-agent.md for full command)
gcloud ai custom-jobs create --region=europe-west4 ...
```

## Architecture

### Inference pipeline (`run.py` + `src/`)

`run.py` is the competition entry point and contains all inference logic (no separate `src/inference.py` or `src/postprocess.py`). Flow: parse args → load models → `collect_images()` → detect → classify → write JSON. All file ops use `pathlib.Path`. All inference is wrapped in `torch.no_grad()`.

Key pipeline stages in `run.py`:
- **WBF ensemble**: `run_ensemble_inference()` runs 3 YOLO models at different resolutions (1280, 640, 640) and merges detections with Weighted Box Fusion.
- **Two-stage classifier**: `classify_crops()` crops each detection, runs through a timm classifier (EfficientNet-B3) with optional TTA (horizontal flip averaging) and classifier ensemble support.
- **Score fusion**: blends YOLO detection confidence with classifier softmax (`SCORE_FUSION_ALPHA`).
- **Prototype matching**: `src/prototype_matcher.py` provides cosine-similarity matching against pre-computed reference embeddings as a fallback for mid-confidence classifier predictions.

`src/constants.py` is the single source of truth for all tuning parameters: confidence thresholds, ensemble weights, classifier config, prototype matching config, and GCP settings. Change thresholds here, not in calling code.

### Training pipeline (`training/`)

`training/train.py` is the Vertex AI job entrypoint. It pulls the dataset from GCS (`gs://ai-nm26osl-1792-nmiai/datasets/`), trains YOLOv8m via ultralytics, logs results to `docs/benchmark_results.md`, then pushes best weights back to GCS. `training/Dockerfile` extends `ultralytics/ultralytics:8.1.0` and adds the `google-cloud-storage` SDK.

`training/data.yaml` configures the YOLO dataset: `nc: 356`, paths to `train/images` and `val/images` under `/workspace/data`.

### Agent ownership

| Agent | Owns |
|-------|------|
| `model-agent` | `training/`, `weights/`, tuning params in `src/constants.py` |
| `inference-agent` | `run.py`, `src/prototype_matcher.py` |
| `qa-agent` | `tests/`, `docs/benchmark_results.md` |
| `lead-agent` | `TASKS.md`, `CLAUDE.md`, `.claude/agents/`, cross-cutting changes |

### MCP

The `nmiai` MCP server (`https://mcp-docs.ainm.no/mcp`) is configured in `.mcp.json` and auto-approved. Use it to discover dataset download endpoints and competition API tools.

### GCP

- Project: `ai-nm26osl-1792`
- Bucket: `gs://ai-nm26osl-1792-nmiai/` (datasets + weights)
- Training container: `us-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest`

## Competition Constraints

### BLOCKED — submission is disqualified if any `.py` file contains:

```python
# Blocked modules
import os          # use pathlib.Path instead
import subprocess
import socket
import ctypes
import builtins

# Blocked builtins
eval(...)
exec(...)
compile(...)
__import__(...)
```

The `security-imports.sh` PostToolUse hook blocks any edit that introduces these. If the hook fires, fix it before continuing. Tests in `tests/test_security.py` also enforce this.

### Submission format

- `run.py` must be at the **root** of the zip (not in a subfolder)
- Output: JSON array — each entry must have exactly these fields:
  ```json
  {"image_id": 42, "category_id": 7, "bbox": [x, y, width, height], "score": 0.95}
  ```
  - `bbox` is `[x_topleft, y_topleft, width, height]` in pixels (xywh, **not** xyxy)
  - `score` must be Python `float`, `category_id` and `image_id` must be Python `int`
  - `image_id` comes from `int(img_path.stem.split("_")[-1])` — filenames are `img_00042.jpg` → `42`
- Max weight files: **420 MB** total, **3 weight files** max
- Timeout: **300 seconds** for the entire test set on an NVIDIA L4 (24 GB VRAM, 8 GB RAM)
- Pre-installed: Python 3.11, ultralytics 8.1.0, PyTorch 2.6.0+cu124, onnxruntime-gpu 1.20.0
- No `pip install` at runtime

## Skills

| Skill | When to use |
|-------|-------------|
| `/validate-submission` | Before every zip — runs full pre-flight checklist |
| `/benchmark-inference` | After any change to run.py or model format — checks 300s budget |

## Key Rules

- All constants (thresholds, paths, limits) go in `src/constants.py` — no magic numbers in logic
- All test output must be piped through `tail` — never use verbose mode
- Inference timing results go to `docs/benchmark_results.md` (Tier 1 summary), never parsed from stdout
- Each agent writes only to their owned files; cross-cutting changes go through lead-agent
- Stage specific files by name before committing — never `git add .`
