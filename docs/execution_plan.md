# Execution Plan: 0.9025 -> 0.9200+

**Created:** 2026-03-20 (evening)
**Deadline:** 2026-03-22 (Sunday evening)
**Submissions:** 3 tonight, 6 Saturday, 3 Sunday = 12 total

---

## Phase 0: Immediate (Do NOW, before classifiers finish)

These tasks have ZERO dependency on training jobs completing.

### Task 0.1: Box Padding for Classifier Crops
- **Status:** CAN DO NOW
- **Expected gain:** +0.002-0.005
- **Time:** 30 minutes
- **Dependencies:** None
- **Agent:** inference-agent

**Files to change:**
- `run.py` — modify `classify_crops()`, add padding around bbox before cropping
- `src/constants.py` — add `CLASSIFIER_CROP_PAD_RATIO = 0.10`

**Code change in `run.py::classify_crops()` (around line 154-162):**
```python
# BEFORE (current):
left = max(0, int(bx))
upper = max(0, int(by))
right = min(img_w, int(bx + bw))
lower = min(img_h, int(by + bh))

# AFTER (with padding):
from src.constants import CLASSIFIER_CROP_PAD_RATIO
pad_w = bw * CLASSIFIER_CROP_PAD_RATIO
pad_h = bh * CLASSIFIER_CROP_PAD_RATIO
left = max(0, int(bx - pad_w))
upper = max(0, int(by - pad_h))
right = min(img_w, int(bx + bw + pad_w))
lower = min(img_h, int(by + bh + pad_h))
```

**Testing:**
- Unit test: add `tests/test_classify.py::test_crop_padding_expands_bbox`
- Verify padded crop is larger than unpadded crop
- Verify clamping to image boundaries works

---

### Task 0.2: Score Fusion (YOLO conf + classifier conf)
- **Status:** CAN DO NOW
- **Expected gain:** +0.001-0.003
- **Time:** 45 minutes
- **Dependencies:** None (works with existing classifier path)
- **Agent:** inference-agent

**Files to change:**
- `src/constants.py` — add `SCORE_FUSION_ALPHA = 0.5` (weight for YOLO conf)
- `run.py` — modify `classify_crops()` to update `pred["score"]` when classifier overrides

**Code change in `run.py::classify_crops()` (around line 190-193):**
```python
# BEFORE:
if conf >= CLASSIFIER_CONFIDENCE_GATE:
    pred["category_id"] = int(cls_id)

# AFTER:
from src.constants import SCORE_FUSION_ALPHA
if conf >= CLASSIFIER_CONFIDENCE_GATE:
    pred["category_id"] = int(cls_id)
    pred["score"] = float(
        SCORE_FUSION_ALPHA * pred["score"] + (1 - SCORE_FUSION_ALPHA) * conf
    )
```

**Testing:**
- Unit test: verify score is blended, not just replaced
- Verify output scores are still valid floats in (0, 1]

---

### Task 0.3: Classifier TTA Module
- **Status:** CAN DO NOW (write code, test when classifier weights exist)
- **Expected gain:** +0.003-0.008
- **Time:** 2 hours
- **Dependencies:** None for code; needs classifier weights to test
- **Agent:** inference-agent

**Files to create:**
- `src/classifier_tta.py` — new module with TTA augmentation logic

**Files to change:**
- `src/constants.py` — add `USE_CLASSIFIER_TTA = True`, `CLASSIFIER_TTA_AUGMENTS = 4`
- `run.py` — import and call TTA variant of classify_crops when enabled

**New module `src/classifier_tta.py`:**
```python
"""Test-time augmentation for the two-stage classifier.

Runs each crop through the classifier multiple times with augmentations,
averages the softmax outputs, and takes argmax for a more robust prediction.
"""

import torch
from torchvision import transforms
from src.constants import CLASSIFIER_INPUT_SIZE

def create_tta_transforms() -> list[transforms.Compose]:
    """Return list of TTA transforms. Each produces a CLASSIFIER_INPUT_SIZE tensor."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    size = CLASSIFIER_INPUT_SIZE
    return [
        # 0: Original (identity)
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 1: Horizontal flip
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 2: Slight scale up + center crop
        transforms.Compose([
            transforms.Resize((int(size * 1.15), int(size * 1.15))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]),
        # 3: Color jitter (deterministic brightness boost)
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]),
    ]

def apply_tta_to_crops(
    crops: list,  # list of PIL.Image
    classifier: torch.nn.Module,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    """Run TTA on a list of PIL crops. Returns (class_ids, confidences).

    For each crop, applies all TTA transforms, runs classifier on all variants,
    averages softmax probabilities, then takes argmax.
    """
    tta_transforms = create_tta_transforms()
    num_augs = len(tta_transforms)
    num_crops = len(crops)

    # Build all augmented tensors: num_crops * num_augs total
    all_tensors = []
    for crop in crops:
        for tfm in tta_transforms:
            all_tensors.append(tfm(crop))

    # Run classifier in batches
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(all_tensors), batch_size):
            batch = torch.stack(all_tensors[i:i + batch_size]).to("cuda")
            logits = classifier(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    # Concatenate all probabilities: shape [num_crops * num_augs, num_classes]
    all_probs_tensor = torch.cat(all_probs, dim=0)

    # Reshape to [num_crops, num_augs, num_classes] and average over augmentations
    all_probs_tensor = all_probs_tensor.view(num_crops, num_augs, -1)
    avg_probs = all_probs_tensor.mean(dim=1)  # [num_crops, num_classes]

    max_probs, class_ids = avg_probs.max(dim=1)
    return class_ids.tolist(), max_probs.tolist()
```

**Changes to `run.py::classify_crops()`:**
```python
# At the top: import conditionally
from src.constants import USE_CLASSIFIER_TTA

# Replace the batch inference block with:
if USE_CLASSIFIER_TTA:
    from src.classifier_tta import apply_tta_to_crops
    # crops is list of PIL images (already prepared above)
    all_class_ids, all_confidences = apply_tta_to_crops(
        pil_crops, classifier, batch_size
    )
else:
    # ... existing batch inference code ...
```

This requires refactoring `classify_crops()` to collect PIL crops separately from tensors. The PIL crops list must be built before the tensor conversion step.

**Testing:**
- `tests/test_classifier_tta.py::test_tta_transforms_output_shape` — verify each transform produces correct tensor size
- `tests/test_classifier_tta.py::test_tta_averaging` — verify averaging over augmentations works correctly with mock model
- Security: verify no banned imports in new module

---

### Task 0.4: Classifier Ensemble Infrastructure
- **Status:** CAN DO NOW (write code, deploy when weights available)
- **Expected gain:** +0.005-0.015
- **Time:** 2 hours
- **Dependencies:** None for code; needs multiple classifier weights to deploy
- **Agent:** inference-agent

**Files to change:**
- `src/constants.py` — add:
  ```python
  CLASSIFIER_ENSEMBLE_WEIGHTS: list[dict[str, str]] = [
      {"path": "weights/classifier.pt", "model_name": "efficientnet_b3", "input_size": 300},
      # {"path": "weights/classifier2.pt", "model_name": "convnext_small.fb_in22k_ft_in1k", "input_size": 384},
  ]
  # When list has >1 entry, ensemble mode is used. Empty list = single classifier mode.
  ```
- `run.py` — modify `load_classifier()` to return a list of classifiers

**Design decision:** Rather than modifying `load_classifier()` return type (breaking change), create a new function:

```python
def load_classifiers() -> list[tuple[torch.nn.Module, int]]:
    """Load classifier ensemble. Returns list of (model, input_size) tuples."""
    from src.constants import CLASSIFIER_ENSEMBLE_WEIGHTS
    ...
```

And modify `classify_crops()` to accept the list and average softmax across models.

**Testing:**
- Mock test with a single classifier verifies ensemble-of-one matches single-model behavior
- Test that softmax averaging produces valid probability distribution

---

### Task 0.5: Prototype Matching Infrastructure
- **Status:** CAN START NOW (write inference code + pre-computation script)
- **Expected gain:** +0.005-0.020
- **Time:** 4 hours
- **Dependencies:** Needs a trained classifier to extract features; can write all code now
- **Agent:** inference-agent (inference code), model-agent (pre-computation script)

**Files to create:**
- `src/prototype_matching.py` — inference-time prototype matching module
- `scripts/compute_prototypes.py` — offline script to extract and save prototype embeddings

**Files to change:**
- `src/constants.py` — add:
  ```python
  USE_PROTOTYPE_MATCHING = False  # Enable when prototypes.pt exists
  PROTOTYPE_PATH = "weights/prototypes.pt"
  PROTOTYPE_CONFIDENCE_THRESHOLD = 0.5  # Use prototypes when classifier conf < this
  PROTOTYPE_ALPHA = 0.3  # Weight for prototype similarity in score fusion
  ```
- `run.py` — add prototype matching call after classifier scoring

**Module `src/prototype_matching.py`:**
```python
"""Reference image prototype matching for low-confidence predictions.

Uses precomputed feature centroids from reference product images to
classify detections via nearest-centroid cosine similarity.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from src.constants import PROTOTYPE_PATH, PROTOTYPE_CONFIDENCE_THRESHOLD, PROTOTYPE_ALPHA


def load_prototypes() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load precomputed prototypes. Returns (centroids, class_ids) or None."""
    p = Path(PROTOTYPE_PATH)
    if not p.exists():
        return None
    data = torch.load(str(p), map_location="cpu")
    return data["centroids"].cuda(), data["class_ids"].cuda()


def extract_features(
    classifier: torch.nn.Module,
    crop_tensors: torch.Tensor,
) -> torch.Tensor:
    """Extract penultimate-layer features from classifier.

    Uses timm's forward_features() to get spatial features,
    then global average pools to get a 1D embedding per crop.
    """
    with torch.no_grad():
        features = classifier.forward_features(crop_tensors)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # GAP
        features = F.normalize(features, dim=1)
    return features


def match_prototypes(
    features: torch.Tensor,
    centroids: torch.Tensor,
    class_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest prototype for each feature vector.

    Returns (matched_class_ids, similarity_scores).
    """
    # Cosine similarity: features [N, D] @ centroids.T [D, K] -> [N, K]
    similarities = features @ centroids.T
    best_sim, best_idx = similarities.max(dim=1)
    matched_classes = class_ids[best_idx]
    return matched_classes, best_sim
```

**Script `scripts/compute_prototypes.py`:**
```python
"""Pre-compute prototype embeddings from reference product images.

Loads a trained classifier, extracts penultimate-layer features for all
reference images, averages per class, and saves as weights/prototypes.pt.

Usage:
    python scripts/compute_prototypes.py \
        --classifier weights/classifier.pt \
        --model-name efficientnet_b3 \
        --data-root training/data \
        --output weights/prototypes.pt
"""
# Implementation: iterate ref images, extract features, group by class,
# average, L2-normalize, save {centroids: [K, D], class_ids: [K]}
```

**Integration in `run.py::main()`:**
```python
# After classify_crops():
if USE_PROTOTYPE_MATCHING:
    from src.prototype_matching import load_prototypes, refine_with_prototypes
    proto_data = load_prototypes()
    if proto_data is not None:
        predictions = refine_with_prototypes(
            image_paths, predictions, classifier, proto_data
        )
```

**Testing:**
- `tests/test_prototype_matching.py::test_cosine_similarity_nearest` — verify matching logic
- `tests/test_prototype_matching.py::test_load_prototypes_missing_file` — returns None gracefully
- Security: verify no banned imports

---

## Phase 1: Deploy Classifier (when weights ready, ~2-3h from now)

### Task 1.1: Download and Deploy Best Classifier
- **Status:** WAIT for training to finish (~2-3h)
- **Expected gain:** +0.005-0.015
- **Time:** 30 minutes
- **Dependencies:** Classifier v1 or v2 training completion
- **Agent:** model-agent

**Actions:**
1. `gsutil cp gs://YOUR_GCS_BUCKET/weights/classifier_efficientnet_b3.pt weights/classifier.pt`
2. Verify `USE_CLASSIFIER = True` in `src/constants.py`
3. Set `CLASSIFIER_CONFIDENCE_GATE = 0.10` (aggressive, since classifier >> YOLO for classification)
4. Run `python run.py --input training/data/yolo/val/images --output /tmp/val_preds.json`
5. Verify timing is within budget

**Testing:**
- Run on val images, check output format
- Time the full pipeline
- Compare prediction distribution with and without classifier

---

### Task 1.2: Submit Sub 1 (Baseline + Classifier)
- **Status:** WAIT for Task 1.1
- **Dependencies:** Task 1.1, and Tasks 0.1 + 0.2 (box padding + score fusion)
- **Agent:** lead-agent

**Actions:**
1. Run `scripts/validate_submission.sh`
2. Run `scripts/create_submission.sh` (or `scripts/create_submission.py`)
3. Submit via competition API
4. Record score

---

### Task 1.3: Tune Classifier Confidence Gate
- **Status:** WAIT for Task 1.1
- **Expected gain:** +0.002-0.005
- **Time:** 30 minutes
- **Dependencies:** Classifier deployed, val set available
- **Agent:** model-agent

**Actions:**
- Test gate values {0.05, 0.10, 0.15, 0.20, 0.30} by running eval_offline with full pipeline
- Pick the one that maximizes val score
- Update `CLASSIFIER_CONFIDENCE_GATE` in `src/constants.py`

---

## Phase 2: Saturday Morning (when overnight training finishes)

### Task 2.1: Evaluate All New Models
- **Status:** WAIT for overnight training jobs
- **Time:** 1-2 hours
- **Dependencies:** Corrected-label YOLO jobs + additional classifier jobs
- **Agent:** qa-agent

**Actions:**
1. Download all completed weights from GCS
2. Run `scripts/eval_offline.py --compare` on each new YOLO model
3. Run classifier accuracy evaluation on each new classifier
4. Log results to `docs/benchmark_results.md`

---

### Task 2.2: Swap Corrected-Label YOLO Models
- **Status:** WAIT for Task 2.1
- **Expected gain:** +0.005-0.015
- **Time:** 1 hour
- **Dependencies:** Corrected-label YOLO training jobs complete
- **Agent:** model-agent

**Files to change:**
- `src/constants.py` — update `ENSEMBLE_WEIGHTS` and `ENSEMBLE_IMAGE_SIZES` with best 3 models

---

### Task 2.3: Deploy Classifier Ensemble
- **Status:** WAIT for multiple classifiers to finish training
- **Expected gain:** +0.005-0.015
- **Time:** 1 hour (code already written in Task 0.4)
- **Dependencies:** Task 0.4 code + 2+ trained classifier weights
- **Agent:** model-agent

**Actions:**
1. Download best 2 classifier weights (different architectures preferred)
2. Update `CLASSIFIER_ENSEMBLE_WEIGHTS` in `src/constants.py`
3. Test timing and accuracy

**Weight budget check:**
- 3 YOLO models: ~221 MB
- 2 classifiers: ~129 MB (EfficientNet-B3 46MB + ConvNeXt-Small 83MB)
- Prototypes: ~2 MB
- Total: ~352 MB (within 420 MB limit)

---

### Task 2.4: Re-sweep WBF/NMS Thresholds
- **Status:** WAIT for Task 2.2
- **Expected gain:** +0.002-0.005
- **Time:** 1 hour
- **Dependencies:** New YOLO models deployed
- **Agent:** model-agent

**Actions:**
- Use `scripts/sweep_thresholds.py` or create a sweep script
- Test `WBF_IOU_THRESHOLD` in {0.45, 0.50, 0.55, 0.60, 0.65}
- Test `WBF_SKIP_BOX_THRESHOLD` in {0.0005, 0.001, 0.005, 0.01}
- Test `IOU_THRESHOLD` (NMS) in {0.40, 0.45, 0.50, 0.55, 0.60}
- Update `src/constants.py` with best values

---

## Phase 3: Saturday Afternoon

### Task 3.1: Deploy Classifier TTA
- **Status:** WAIT for Task 0.3 code + Task 1.1 weights
- **Expected gain:** +0.003-0.008
- **Time:** 30 minutes to integrate (code written in Task 0.3)
- **Dependencies:** Task 0.3 complete, classifier weights available
- **Agent:** inference-agent

**Actions:**
1. Set `USE_CLASSIFIER_TTA = True` in `src/constants.py`
2. Run timing test — verify TTA adds acceptable overhead
3. Run on val set, compare score with and without TTA

---

### Task 3.2: Compute and Deploy Prototypes
- **Status:** WAIT for Task 0.5 code + Task 1.1 weights
- **Expected gain:** +0.005-0.020
- **Time:** 1 hour
- **Dependencies:** Task 0.5 code, trained classifier, reference images
- **Agent:** model-agent

**Actions:**
1. Run `python scripts/compute_prototypes.py --classifier weights/classifier.pt`
2. Verify `weights/prototypes.pt` size (~2 MB)
3. Set `USE_PROTOTYPE_MATCHING = True` in `src/constants.py`
4. Test on val set — verify improvement on rare classes

---

### Task 3.3: Submit Sub 4-6 (Saturday batch)
- **Agent:** lead-agent

**Sub 4:** Best corrected YOLO + best single classifier + box padding + score fusion
**Sub 5:** Add classifier ensemble (2 models)
**Sub 6:** Add prototype matching

---

## Phase 4: Saturday Evening

### Task 4.1: Category-Aware Post-Processing
- **Status:** CAN START after Phase 1 (needs val predictions to analyze)
- **Expected gain:** +0.002-0.008
- **Time:** 3-4 hours
- **Dependencies:** Working pipeline with classifier
- **Agent:** inference-agent

**Files to create:**
- `src/postprocess.py` — category-aware NMS and confidence calibration

**Module `src/postprocess.py`:**
```python
"""Category-aware post-processing for detection predictions.

Includes:
1. Category-aware NMS: same-class nearby boxes kept, different-class overlaps resolved
2. Per-class confidence calibration (loaded from precomputed weights)
"""

import torch
from src.constants import NUM_CLASSES


def category_aware_nms(
    predictions: list[dict],
    same_class_iou_threshold: float = 0.3,
    diff_class_iou_threshold: float = 0.5,
) -> list[dict]:
    """Apply category-aware NMS to predictions.

    Rules:
    - Two boxes with SAME category and IoU > same_class_iou_threshold: keep both
      (adjacent same-product on shelf)
    - Two boxes with DIFFERENT categories and IoU > diff_class_iou_threshold:
      keep only higher confidence one
    """
    ...


def apply_confidence_calibration(
    predictions: list[dict],
    calibration_weights: dict[int, float],
) -> list[dict]:
    """Scale each prediction's score by a per-class calibration weight."""
    for pred in predictions:
        cls_id = pred["category_id"]
        if cls_id in calibration_weights:
            pred["score"] = float(pred["score"] * calibration_weights[cls_id])
    return predictions
```

**Files to change:**
- `src/constants.py` — add:
  ```python
  USE_CATEGORY_AWARE_NMS = False
  CATEGORY_NMS_SAME_CLASS_IOU = 0.3
  CATEGORY_NMS_DIFF_CLASS_IOU = 0.5
  CALIBRATION_WEIGHTS_PATH = "weights/calibration.json"
  ```
- `run.py` — add post-processing call after classify_crops

**Pre-computation needed:**
- `scripts/compute_calibration.py` — run full pipeline on val set, compute per-class optimal weights

**Testing:**
- `tests/test_postprocess.py::test_category_nms_keeps_same_class`
- `tests/test_postprocess.py::test_category_nms_removes_diff_class_overlap`
- `tests/test_postprocess.py::test_calibration_scales_scores`

---

### Task 4.2: Per-Class Confidence Calibration
- **Status:** WAIT for pipeline producing val predictions
- **Expected gain:** +0.002-0.005
- **Time:** 2 hours
- **Dependencies:** Working pipeline, val ground truth
- **Agent:** model-agent

**Files to create:**
- `scripts/compute_calibration.py` — analyze val predictions, compute optimal per-class weights, save to `weights/calibration.json`

**Integration:** Already handled by `src/postprocess.py` from Task 4.1

---

### Task 4.3: Submit Sub 7-9 (Saturday evening batch)
- **Agent:** lead-agent

**Sub 7:** Best config + tuned WBF/NMS thresholds
**Sub 8:** Full pipeline: YOLO ensemble + classifier ensemble + TTA + prototypes + post-processing
**Sub 9:** Alternative config (e.g., 2 YOLO + 2 classifiers + aggressive prototype matching)

---

## Phase 5: Sunday (Final)

### Task 5.1: Final Tuning
- **Time:** 2-3 hours
- **Agent:** model-agent + qa-agent

**Actions:**
- Analyze Saturday submission results
- Fine-tune the best-performing config
- Adjust any thresholds based on leaderboard feedback

### Task 5.2: Final Submissions (Sub 10-12)
- **Agent:** lead-agent

**Sub 10:** Fine-tuned best config
**Sub 11:** Safety resubmit of best-ever scoring config
**Sub 12:** Hail Mary — most aggressive config with all tricks enabled

---

## Dependency Graph

```
Phase 0 (NOW, parallel):
  0.1 Box Padding ----\
  0.2 Score Fusion ----+---> Phase 1 (when classifier ready)
  0.3 Classifier TTA --/      |
  0.4 Ensemble Infra -/       v
  0.5 Prototype Infra        1.1 Deploy Classifier
                               |
                               +-> 1.2 Submit Sub 1
                               +-> 1.3 Tune Gate
                               |
                        [Overnight training completes]
                               |
                               v
                        Phase 2 (Sat morning):
                          2.1 Evaluate All Models
                            |
                            +-> 2.2 Swap YOLO Models
                            +-> 2.3 Deploy Classifier Ensemble
                            +-> 2.4 Re-sweep Thresholds
                            |
                            v
                        Phase 3 (Sat afternoon):
                          3.1 Deploy TTA
                          3.2 Deploy Prototypes
                          3.3 Submit Sub 4-6
                            |
                            v
                        Phase 4 (Sat evening):
                          4.1 Category-Aware Post-Processing
                          4.2 Per-Class Calibration
                          4.3 Submit Sub 7-9
                            |
                            v
                        Phase 5 (Sunday):
                          5.1 Final Tuning
                          5.2 Submit Sub 10-12
```

---

## File Ownership Summary

| File | Owner | Phase |
|------|-------|-------|
| `src/constants.py` | model-agent | All phases (constants only) |
| `run.py` | inference-agent | Phase 0-1 |
| `src/classifier_tta.py` (NEW) | inference-agent | Phase 0 |
| `src/prototype_matching.py` (NEW) | inference-agent | Phase 0 |
| `src/postprocess.py` (NEW) | inference-agent | Phase 4 |
| `scripts/compute_prototypes.py` (NEW) | model-agent | Phase 3 |
| `scripts/compute_calibration.py` (NEW) | model-agent | Phase 4 |
| `tests/test_classifier_tta.py` (NEW) | qa-agent | Phase 0 |
| `tests/test_prototype_matching.py` (NEW) | qa-agent | Phase 0 |
| `tests/test_postprocess.py` (NEW) | qa-agent | Phase 4 |

---

## Constants to Add to `src/constants.py`

All in one block, added at the end of the file:

```python
# ---------------------------------------------------------------------------
# Classifier enhancements
# ---------------------------------------------------------------------------

# Box padding: expand YOLO bbox by this ratio before cropping for classifier
CLASSIFIER_CROP_PAD_RATIO = 0.10

# Score fusion: blend YOLO detection conf with classifier conf
# final_score = alpha * yolo_conf + (1 - alpha) * classifier_conf
SCORE_FUSION_ALPHA = 0.5

# Classifier TTA: run multiple augmented versions and average softmax
USE_CLASSIFIER_TTA = False  # Enable after testing timing impact

# Classifier ensemble: load multiple classifiers and average predictions
# Each entry: {"path": str, "model_name": str, "input_size": int}
# Empty list = use single classifier from CLASSIFIER_PATH
CLASSIFIER_ENSEMBLE_WEIGHTS: list[dict] = []

# ---------------------------------------------------------------------------
# Prototype matching
# ---------------------------------------------------------------------------

USE_PROTOTYPE_MATCHING = False  # Enable when prototypes.pt is computed
PROTOTYPE_PATH = "weights/prototypes.pt"
PROTOTYPE_CONFIDENCE_THRESHOLD = 0.5  # Use prototypes when classifier conf < this
PROTOTYPE_ALPHA = 0.3  # Blend weight for prototype similarity

# ---------------------------------------------------------------------------
# Category-aware post-processing
# ---------------------------------------------------------------------------

USE_CATEGORY_AWARE_NMS = False
CATEGORY_NMS_SAME_CLASS_IOU = 0.3
CATEGORY_NMS_DIFF_CLASS_IOU = 0.5
CALIBRATION_WEIGHTS_PATH = "weights/calibration.json"
```

---

## run.py Main Flow (after all changes)

```python
def main():
    # Parse args, setup paths
    image_paths = collect_images(input_dir)

    # Load models
    classifiers = load_classifiers()  # list of (model, input_size)
    proto_data = load_prototypes() if USE_PROTOTYPE_MATCHING else None

    # Stage 1: Detection (YOLO ensemble + WBF)
    if ENSEMBLE_WEIGHTS:
        models = load_ensemble_models()
        predictions = run_ensemble_inference(models, image_paths)
    else:
        model = load_model()
        predictions = run_inference(model, image_paths)

    # Stage 2: Classification (classifier ensemble + TTA + prototypes)
    if classifiers:
        predictions = classify_crops(image_paths, predictions, classifiers)

    # Stage 3: Prototype refinement (for low-confidence classifier predictions)
    if proto_data is not None:
        predictions = refine_with_prototypes(image_paths, predictions, classifiers[0], proto_data)

    # Stage 4: Post-processing
    if USE_CATEGORY_AWARE_NMS:
        predictions = category_aware_nms(predictions)
    if Path(CALIBRATION_WEIGHTS_PATH).exists():
        predictions = apply_confidence_calibration(predictions)

    # Write output
    output_path.write_text(json.dumps(predictions))
```

---

## Checklist for Each Change

Before merging ANY change:
1. [ ] `ruff check src/ run.py` passes
2. [ ] `ruff format src/ run.py` passes
3. [ ] `python -m pytest tests/ -q --tb=line -m "not slow" 2>&1 | tail -20` passes
4. [ ] No banned imports (`import os`, `subprocess`, etc.)
5. [ ] All constants in `src/constants.py`, no magic numbers
6. [ ] `scripts/validate_submission.sh` passes
7. [ ] Total weight files < 420 MB

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Classifier makes things worse | Confidence gate prevents bad overrides; test on val first |
| TTA too slow | Profile timing; can disable via `USE_CLASSIFIER_TTA = False` |
| Prototype matching hurts | Only applies to low-confidence predictions; threshold tunable |
| Weight budget exceeded | Pre-calculate sizes; drop YOLOv8m if needed |
| Code gets messy | Each feature in its own module under `src/`; feature flags in constants |
| Sunday submission pressure | Always keep best-ever config on a named git branch |

---

## What to Do RIGHT NOW

1. **inference-agent:** Implement Tasks 0.1 (box padding) and 0.2 (score fusion) — these are tiny, high-value changes to `run.py` and `src/constants.py`
2. **inference-agent:** Start Task 0.3 (classifier TTA module) — write `src/classifier_tta.py`
3. **inference-agent:** Start Task 0.4 (classifier ensemble) — refactor `load_classifier()` / `classify_crops()` to support multiple models
4. **qa-agent:** Write tests for Tasks 0.1-0.4 as they complete
5. **model-agent:** Monitor training jobs, prepare to download weights as soon as they finish
6. **lead-agent:** Prepare submission workflow; verify `scripts/validate_submission.sh` works
