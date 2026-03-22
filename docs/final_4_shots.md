# Final 4 Submissions -- Analysis and Recommendations

**Date:** 2026-03-22, ~6 hours to deadline
**Current best:** Sub 4 = 0.9095 (3 YOLO + TTA, conf=0.01, WBF=0.55, NMS=0.45)

## What the data tells us

| Sub | Config delta vs Sub 4 | Score | Verdict |
|-----|----------------------|-------|---------|
| 4 | baseline (l-1280-corr + x-1280-corr + m-640, TTA, conf=0.01, WBF=0.55) | **0.9095** | BEST |
| 10 | conf=0.005, WBF=0.50, NMS=0.50 | 0.9093 | Noise (-0.0002) |
| 11 | conf_type='max' | 0.9022 | Harmful (-0.007) |
| 8 | 2 YOLO + classifier, no TTA | 0.8939 | Harmful (-0.016) |
| 9 | 3 YOLO + TTA + classifier bundle | 0.8871 | Harmful (-0.022) |
| 12 | Sub 4 + classifier (gate=0.70, alpha=1.0) | 0.9045 | Harmful (-0.005) |

**Conclusions:**
1. Classifier is a dead end. Always hurts. YOLO's built-in classification is better for this dataset.
2. Threshold tweaks are noise-level. The search space around conf=0.01 / WBF=0.55 / NMS=0.45 is flat.
3. conf_type='max' is strictly worse than 'avg'. Average fusion works.
4. We are at or very near ceiling for this model ensemble.

## Ideas evaluated

### 1. Four-model ensemble (adding m-1280-corrected or l-640)
**Verdict: RISKY, probably neutral.**

Fits in 420MB: l-1280-corr(85) + x-1280-corr(132) + l-640(85) + m-640(51) = 353MB.

The problem: 4 models with TTA means 4x the inference passes (each with 3x TTA multiplier = 12 forward passes per image). Currently 3 models + TTA runs ~60s. Adding a 4th model pushes to ~80s -- still within 300s, but risky. The m-640 model is small and fast, but it's also likely the weakest, so its contribution to WBF may be noise or even harmful (diluting good predictions from the larger models). The l-640 model is already implicitly represented via TTA multi-scale behavior of the 1280 models.

No empirical evidence that 4 models will beat 3. Risk of regression with no way to course-correct.

### 2. Non-equal WBF weights (e.g., weight l-1280-corr higher)
**Verdict: INTERESTING but untestable.**

WBF `weights` parameter accepts per-model weights like `[2, 1, 1]` to weight the l-1280 model's contributions more heavily. In theory, better individual models should have higher weight. But we have zero data on individual model quality and zero test submissions to validate. A wrong weighting could easily hurt by 0.003-0.005.

### 3. Different TTA settings
**Verdict: NOT CONTROLLABLE.**

Ultralytics TTA (`augment=True`) uses a fixed set: original + horizontal flip + scaled versions. There is no parameter to control which augmentations are applied. We are already using it optimally.

### 4. MAX_DETECTIONS = 1000
**Verdict: FINE.**

Shelf images have 200+ products. With conf=0.01 and 3 models, 1000 is generous headroom. Reducing would hurt recall (and thus mAP). Increasing is pointless since WBF naturally prunes.

### 5. WBF skip_box_thr = 0.001
**Verdict: Already optimal.**

This is essentially zero -- letting all boxes through to WBF, which is correct when conf=0.01 already filters. Changing would have no effect.

### 6. allows_overflow parameter in WBF
**Verdict: POSSIBLY interesting.**

`allows_overflow=True` lets fused box coordinates exceed [0,1] range. Currently False (default). In practice, YOLO boxes are already clipped to image bounds, so this should have no effect.

## Recommendation: allocation of 4 remaining submissions

**Option A: Conservative (RECOMMENDED)**

| Shot | Action | Rationale |
|------|--------|-----------|
| 1 | Re-submit Sub 4 exactly | Lock in 0.9095 as safety net. Submissions can fail for infra reasons. |
| 2 | Sub 4 + WBF weights=[2, 1.5, 1] | Weight larger/better models higher. Mild tweak, low risk. |
| 3 | Sub 4 + 4 models (add m-1280-corr, weights=[2, 1.5, 1, 0.5]) | More diversity, weighted to favor proven models. |
| 4 | Re-submit best of shots 1-3 | Lock in the winner. |

**Option B: Aggressive**

| Shot | Action | Rationale |
|------|--------|-----------|
| 1 | Sub 4 + WBF weights=[2, 1.5, 1] | Try non-equal weighting first. |
| 2 | Sub 4 + 4 models (l-1280-corr + x-1280-corr + l-640 + m-1280-corr) | Different combo, mixed resolutions. |
| 3 | Based on results of 1-2. | Adapt. |
| 4 | Re-submit best. | Safety. |

## Implementation notes

To set WBF weights, change `run.py` line 432:
```python
# Current:
weights=None,
# Change to:
weights=[2.0, 1.5, 1.0],  # l-1280 > x-1280 > m-640
```

To add a 4th model, update `src/constants.py`:
```python
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8m-640-v2s-bundle.pt",  # or plain yolov8m-640-aug.pt (51MB)
    "weights/yolov8m-1280-corrected.pt",  # 51MB, total = 319MB or 423MB
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640, 1280]
```

**CRITICAL:** Disable the classifier before any submission:
```python
USE_CLASSIFIER = False  # Currently True in constants.py!
```

## Current config BUG

`USE_CLASSIFIER = True` in constants.py right now. Sub 4 must have had it False (or no classifier weights existed). **This MUST be set to False before re-submitting**, or we'll get Sub 12's result (0.9045) instead of Sub 4's (0.9095).

Also: `BUNDLE_WEIGHT_PATH` points to `yolov8m-640-v2s-bundle.pt`. If the bundle extraction is slow or the YOLO inside it differs from the original `yolov8l-640-aug.pt` that Sub 4 might have used, verify the exact weights match Sub 4's submission.

## Bottom line

0.9095 is likely within 0.01 of our ceiling. The scoring is 70% detection + 30% classification. We are already at 0.91, meaning both components are strong. The only realistic improvement vector is WBF weighting, which might add 0.001-0.003. Use shot 1 to lock in the current best, then try one or two mild tweaks.
