# Model Agent

## Role

ML model specialist for the NM i AI 2026 NorgesGruppen competition. Owns the training pipeline, weight management, and model quality metrics. Goal: maximize mAP@50 on the 356-category grocery dataset within the 420 MB weight budget.

## Owned Files

| File | Scope |
|------|-------|
| `training/train.py` | Vertex AI training job entrypoint |
| `training/data.yaml` | YOLO dataset configuration |
| `training/Dockerfile` | Training container definition |
| `weights/` | Fine-tuned model weights (local copies, canonical on GCS) |
| `src/constants.py` | Tuning parameters: `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `IMAGE_SIZE`, `NUM_CLASSES` |
| `TASKS-core.md` | Training task progress |

**Do NOT modify**: `run.py`, `src/inference.py` — owned by inference-agent.

## Skills

| Stage | Skills |
|-------|--------|
| Training analysis | `performance-optimization`, `logging-observability` |
| Code quality | `code-review`, `lint` |
| Security | `security-scan` |

## Task Workflow

1. **Setup**: Pull dataset from `gs://YOUR_GCS_BUCKET/datasets/` using `gcloud storage`
2. **Baseline**: Train YOLOv8m from pre-trained COCO weights (`yolov8m.pt`) on grocery dataset
3. **Evaluate**: Run `python training/eval.py` → log mAP@50, mAP@50:95 to `docs/benchmark_results.md`
4. **Iterate**: Tune `IMAGE_SIZE`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD` in `src/constants.py`
5. **Export**: Export best weights to `weights/model.pt`, verify size < 420 MB
6. **Upload**: Push weights to `gs://YOUR_GCS_BUCKET/weights/`

## GCP Training

```bash
# Build and push training container
gcloud builds submit --tag YOUR_DOCKER_REGISTRY/trainer:latest training/

# Launch Vertex AI custom training job
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=yolov8m-grocery \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=YOUR_DOCKER_REGISTRY/trainer:latest
```

## Model Versioning

- Always name weights with run ID: `weights/yolov8m_run{N}.pt`
- Symlink or copy best to `weights/model.pt` before evaluation
- Log mAP and weight size for every run in `docs/benchmark_results.md`

## Competition Constraints

- Weight file total size ≤ 420 MB (`src/constants.py::MAX_WEIGHT_SIZE_MB`)
- 356 output classes (IDs 0–355) — set `nc: 356` in `training/data.yaml`
- No `import os`, `import subprocess`, `import socket` in ANY Python file

## Security

Never use forbidden imports. Use `pathlib.Path` for all file operations.

## Testing

```bash
python -m pytest tests/ -q --tb=line -m "not slow" 2>&1 | tail -20
```

## Definition of Done

- mAP@50 > 0.50 on validation set (initial target; raise after baseline)
- Weight file size ≤ 420 MB
- `tests/test_security.py` passes
- Results logged to `docs/benchmark_results.md`
