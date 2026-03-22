# LAST 3 SHOTS -- 3 Hours Left

**Date:** 2026-03-22, 3 hours to deadline
**Current best:** 0.9095 (Sub 4)
**Leader:** ~0.9255. **Gap:** 0.016.
**Submissions remaining:** 3

## What we know

| Sub | Config | Score | Delta |
|-----|--------|-------|-------|
| 4 | l-1280-corr + x-1280-corr + l-640-aug, TTA, conf=0.01, WBF=0.55, NMS=0.45, **equal weights** | **0.9095** | -- |
| 10 | conf=0.005, WBF=0.50, NMS=0.50 | 0.9093 | -0.0002 |
| ?? | WBF weights=[2, 1.5, 1] | 0.9088 | -0.0007 |
| 11 | conf_type='max' | 0.9022 | -0.007 |
| 12 | + classifier (gate=0.70) | 0.9045 | -0.005 |
| 8 | 2 YOLO + classifier | 0.8939 | -0.016 |
| 9 | 3 YOLO + classifier bundle | 0.8871 | -0.022 |

**Dead ends:** classifier (always hurts), conf_type='max', non-equal WBF weights, threshold micro-tuning.

## CRITICAL: Fix before ANY submission

Current constants.py has `WBF_MODEL_WEIGHTS = [2.0, 1.5, 1.0]` -- this is NOT the Sub 4 config.
Sub 4 used equal weights (None). **MUST set `WBF_MODEL_WEIGHTS = None` for shots 1 and 2.**

## The 3 shots, ranked by likelihood of improvement

---

### SHOT 1 (HIGHEST CONFIDENCE): 4-model ensemble with equal weights

**Rationale:** This is the ONLY structural change we haven't tried. Adding a 4th model
increases ensemble diversity, which is the primary driver of WBF improvement. The m-1280-corrected
model (50MB) adds a different architecture scale (medium vs large/xlarge) at the same high
resolution. Total = 349.5MB, well under 420MB. With TTA on 4 models, estimated ~80s runtime
(well within 300s). Equal weights because non-equal already proven to hurt.

**Why m-1280-corr specifically:** Same 1280px resolution as the two best models (avoids
resolution mismatch artifacts). Medium architecture provides genuinely different feature
representations from large and xlarge. The corrected labels match the other two 1280 models.

```python
# src/constants.py changes:
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8l-640-aug.pt",
    "weights/yolov8m-1280-corrected.pt",
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640, 1280]
WBF_MODEL_WEIGHTS = None  # Equal weights (proven best)
USE_CLASSIFIER = False
```

**Expected:** +0.001 to +0.005. More models = better WBF fusion = fewer missed boxes and
more stable class votes. Risk is low: worst case is noise-level change like Sub 10.

---

### SHOT 2: Original (non-corrected) models instead of corrected

**Rationale:** This is a genuinely untested hypothesis. The "corrected" models were trained on
cleaned labels but the user noted "corrected models were slightly worse on detection metrics."
The scoring is 70% detection + 30% classification. If the original models have better detection
mAP, that 70% weight could outweigh any classification loss. We have never submitted with
original models.

The original l-1280-aug (84.2MB) and x-1280-aug (137.5MB -- wait, we need to check this
fits). l-1280-aug(84.2) + x-1280-aug(131.0 -- actually checking: the -aug file is 137.4MB
for x). Let's check: 84.2 + 131.0 + 84.1 = nope, x-1280-aug is 131MB based on ls output.

Actually from ls: yolov8x-1280-aug.pt = 137,492,009 bytes = 131.1MB. Same as corrected.

```python
# src/constants.py changes:
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-aug.pt",       # ORIGINAL, not corrected
    "weights/yolov8x-1280-corrected.pt",  # Keep corrected (x may not have original issue)
    "weights/yolov8l-640-aug.pt",
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640]
WBF_MODEL_WEIGHTS = None
USE_CLASSIFIER = False
```

**Alternative (if Shot 1 scores well):** Use Shot 1's 4-model config but swap l-1280-corr
for l-1280-aug (original). This tests whether original detection quality + 4-model diversity
compounds.

**Expected:** +0.001 to +0.003 if detection metrics are indeed better on original labels.
Risk: could be -0.002 if corrected labels actually help classification enough to offset.

---

### SHOT 3: Adaptive -- based on Shot 1 and Shot 2 results

**If Shot 1 > 0.9095 (4 models helped):**
Try 5-model ensemble: l-1280-corr + x-1280-corr + l-640-aug + m-1280-corr + m-640-aug.
Total = 399.5MB (under 420MB). Even more diversity. Use equal weights.

```python
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8l-640-aug.pt",
    "weights/yolov8m-1280-corrected.pt",
    "weights/yolov8m-640-aug.pt",
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640, 1280, 640]
WBF_MODEL_WEIGHTS = None
```

**If Shot 1 <= 0.9095 AND Shot 2 > 0.9095 (originals helped):**
Go all-original: l-1280-aug + x-1280-aug + l-640-aug, equal weights.

**If both <= 0.9095 (nothing helped):**
Re-submit exact Sub 4 config to lock in 0.9095 as final score. Do NOT experiment further.
The 0.9095 is real and reproducible. Protect it.

```python
# Exact Sub 4 config:
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8l-640-aug.pt",
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640]
WBF_MODEL_WEIGHTS = None
USE_CLASSIFIER = False
USE_TTA = True
CONFIDENCE_THRESHOLD = 0.01
WBF_IOU_THRESHOLD = 0.55
IOU_THRESHOLD = 0.45
```

---

## Execution checklist

For EVERY submission:
- [ ] `USE_CLASSIFIER = False`
- [ ] `WBF_MODEL_WEIGHTS` set correctly (None for shots 1, 2; adaptive for 3)
- [ ] `BUNDLE_WEIGHT_PATH = ""`
- [ ] `USE_TTA = True`
- [ ] `CONFIDENCE_THRESHOLD = 0.01`
- [ ] `WBF_IOU_THRESHOLD = 0.55`
- [ ] `IOU_THRESHOLD = 0.45`
- [ ] `HALF_PRECISION = True`
- [ ] `WBF_SKIP_BOX_THRESHOLD = 0.001`
- [ ] `MAX_DETECTIONS_PER_IMAGE = 1000`
- [ ] Run `bash scripts/validate_submission.sh` before uploading
- [ ] Verify total weight size < 420MB
- [ ] Time between submissions: wait for score before sending next

## Why NOT these ideas

| Idea | Why skip |
|------|----------|
| HALF_PRECISION=False | FP32 is 2x slower, could timeout. No evidence it helps mAP. |
| TTA off | Sub 4 had TTA on. Removing it would lose ~1-3% accuracy. |
| WBF_SKIP_BOX_THRESHOLD=0.0001 | With conf=0.01 already filtering, boxes below 0.001 are noise. No effect. |
| MAX_DETECTIONS=2000 | 1000 already generous. More detections = more noise in WBF. |
| Different ENSEMBLE_IMAGE_SIZES | Running 1280 models at 640 loses resolution. Running 640 at 1280 wastes time. |
| x-640-aug in ensemble | x-640 is 131MB for a 640px model -- l-640 is better value at 84MB. |
