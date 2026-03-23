# NM i AI 2026 -- NorgesGruppen Grocery Detection

Object detection and classification of grocery products on store shelves, built for the [NM i AI 2026](https://www.nmiai.no/) competition hosted by NorgesGruppen.

**Final score: 0.9121 mAP | Rank #61 out of 349 teams**

## Competition Overview

The task was to detect and classify grocery products from 356 categories in shelf images. Scoring was a weighted combination of detection mAP (70%, category-agnostic IoU >= 0.5) and classification mAP (30%, correct category + IoU >= 0.5). Submissions ran on an NVIDIA L4 GPU with a 300-second timeout and a 420 MB weight limit.

## Approach

### 4-Model WBF Ensemble with TTA

The final solution uses a multi-model ensemble fused with Weighted Box Fusion (WBF):

1. **YOLOv8l @ 1280px** -- trained on corrected labels
2. **YOLOv8x @ 1280px** -- trained with augmentation
3. **YOLOv8l @ 640px + YOLOv8x @ 1280px** -- packed in a dual bundle weight file

All models run with Test-Time Augmentation (TTA). Detections are merged using WBF (IoU threshold 0.55) to combine complementary predictions from different model scales and architectures.

A two-stage EfficientNet-B3 classifier was developed for category refinement, and prototype matching via cosine similarity was explored as a fallback. Both are included in the codebase but were not active in the final submission -- the YOLO ensemble alone achieved the best score.

### The Journey: 0.7084 to 0.9121 in 3 Days

| Submission | Score | What changed |
|-----------|-------|-------------|
| Sub 1 | 0.7084 | Baseline YOLOv8m @ 640px |
| Sub 2 | 0.8033 | Lower confidence threshold (0.01), higher max detections |
| Sub 3 | 0.8211 | YOLOv8l @ 1280px |
| Sub 4 | 0.8685 | 3-model WBF ensemble + TTA |
| Sub 11 | 0.9042 | Corrected training labels, retrained models |
| Sub 12 | 0.9121 | 4-model ensemble via dual bundle packing |

## Repository Structure

```
run.py                  # Competition entry point -- all inference logic
src/
  constants.py          # All tuning parameters (thresholds, paths, model config)
  prototype_matcher.py  # Cosine similarity matching against reference embeddings
training/
  train.py              # Vertex AI training entrypoint (YOLOv8 via ultralytics)
  train_classifier.py   # Two-stage classifier training (timm/EfficientNet)
  Dockerfile            # Training container
  data.yaml             # YOLO dataset config (356 classes)
  vertex-job-*.yaml     # Vertex AI job configs for various model/resolution combos
tests/                  # Test suite (security checks, output format, constants)
scripts/                # Dataset download, submission validation, prototype precomputation
docs/                   # Strategy documents, benchmark logs, training monitor logs
weights/                # Model weight files (not included in repo -- see below)
```

## Running Inference

```bash
python run.py --input /path/to/images --output predictions.json
```

Output is a JSON array where each entry has:
```json
{"image_id": 42, "category_id": 7, "bbox": [x, y, width, height], "score": 0.95}
```

Weight files are not included in this repository due to size. The expected weights are:
- `weights/yolov8l-1280-corrected.pt`
- `weights/yolov8x-1280-aug.pt`
- `weights/yolov8-mixed-bundle.pt`

## Training

Training was done on Google Cloud Vertex AI using the configs in `training/`. See `training/train.py` for the YOLOv8 training pipeline and `training/train_classifier.py` for the two-stage classifier.

Dataset was hosted on GCS at `gs://YOUR_GCS_BUCKET/datasets/`.

## Requirements

- Python 3.11
- ultralytics 8.1.0
- PyTorch 2.6.0+cu124
- timm 0.9.12
- torchvision
- Pillow
- ensemble-boxes (for WBF)

## License

MIT -- see [LICENSE](LICENSE).
