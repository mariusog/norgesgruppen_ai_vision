# FINAL PLAN: 3 Submissions, 24 Hours, 0.9095 -> Top 10

**Date:** Saturday March 21, 2026, evening
**Deadline:** Sunday March 22, 15:00 Oslo time
**Current best:** 0.9095 (Sub 4, rank ~#23)
**Leader:** 0.9255 (Fenrir's byte, 14 submissions)
**Gap to #1:** 0.016 | **Gap to top 10:** ~0.011
**Submissions remaining:** 3

---

## 1. ANATOMY OF THE 0.016 GAP

### Score formula
```
Score = 0.7 * det_mAP@0.5 + 0.3 * cls_mAP@0.5
```

Detection mAP: IoU >= 0.5, category ignored (just "did you find a box?").
Classification mAP: IoU >= 0.5 AND correct category_id.

### Decomposing our 0.9095

The detection-only baseline scores "up to 0.70" (per competition docs). This means
if a team got perfect detection but random classification, they would score 0.70.
Our 0.9095 is well above this, meaning our YOLO ensemble is providing decent
classification too.

Working backwards from plausible numbers:

| Scenario | det_mAP | cls_mAP | Score |
|----------|---------|---------|-------|
| **Us (0.9095)** | ~0.955 | ~0.803 | 0.9095 |
| **Leader (0.9255)** | ~0.965 | ~0.850 | 0.9255 |
| **Gap** | ~0.010 | ~0.047 | 0.016 |
| **Gap contribution** | 0.007 (70%) | 0.014 (30%) | |

**Key insight:** The gap is roughly 0.007 from detection and 0.009 from classification.
That is a 44/56 split. Classification improvement has the higher marginal return
because we have more room to grow there (+0.047 potential vs +0.010).

### What does +0.01 det_mAP mean?
Finding ~1% more objects on dense shelves. The top teams likely use:
- Better WBF fusion parameters tuned on validation
- More diverse ensemble members (different architectures, not just YOLO size variants)
- Possibly 4+ model ensembles bundled cleverly
- Lower confidence thresholds with better post-processing

### What does +0.047 cls_mAP mean?
The top teams almost certainly use a two-stage classifier. YOLO's built-in
classification head is mediocre for 356 fine-grained grocery categories.
A dedicated classifier at 91% accuracy vs YOLO's ~75-80% is the difference.

**Bottom line: The classifier is where the points are. We MUST get it working.**

---

## 2. CLASSIFIER WITH ALL BUGS FIXED: IMPACT ESTIMATE

### Bugs found and fixed (all in constants.py):
1. SCORE_FUSION_ALPHA: 0.7 -> 1.0 (keep YOLO scores untouched)
2. CLASSIFIER_INPUT_SIZE: 300 -> 384 (match training resolution)
3. CLASSIFIER_CONFIDENCE_GATE: 0.15 -> 0.70 (only override when confident)
4. USE_CLASSIFIER_TTA: True -> False (h-flip corrupts text on grocery products)

### Expected impact calculation

With all bugs fixed and SCORE_FUSION_ALPHA=1.0:
- Detection mAP: IDENTICAL to Sub 4 (0.9095 baseline). No score corruption.
- Classification: Swin-Tiny at 91% accuracy overrides YOLO on ~60% of detections
  (those above the 0.70 confidence gate).
- YOLO classification accuracy: ~75-80% (estimated from score decomposition)
- On the 60% overridden: accuracy goes from ~75% to ~91% = +16% improvement on 60% = +9.6% overall
- cls_mAP improvement: roughly +0.05 to +0.08 (optimistic)
- Score gain: +0.05 * 0.3 = +0.015 to +0.08 * 0.3 = +0.024

**Conservative estimate for Sub 1: +0.008 to +0.015 -> score 0.917 to 0.925**

This alone could land us in top 10.

### Critical risk: the classifier might STILL hurt
If YOLO's classification is better than we think (e.g., 85% on test distribution),
then even a 91% classifier with a 0.70 gate adds only marginal improvement. But with
SCORE_FUSION_ALPHA=1.0, the DOWNSIDE is capped: worst case, detection mAP stays
identical and classification gets slightly worse on the gated predictions.

**Risk is asymmetric and favorable. This is the right Sub 1.**

---

## 3. WEIGHT BUDGET ANALYSIS (CORRECTED NUMBERS)

**Hard constraint: max 3 weight files, max 420 MB total.**

The professor_review_sat.md stated Swin-Tiny is 28MB. **THIS IS WRONG.**
The actual file `classifier_swin_tiny_224.pt` is **107 MB** (not 28MB).

### Actual classifier sizes:
| Classifier | Size | Val Accuracy |
|-----------|------|-------------|
| EffNet-B3 (classifier.pt) | 44 MB | ~90% |
| EffNet-B3 focal (384px) | 44 MB | ~88% |
| EfficientNetV2-S (384px) | 80 MB | ~91% |
| Swin-Tiny (224px) | 107 MB | ~91% |
| ConvNeXt-Small (384px) | 190 MB | ~91% |

### Configs that fit (3 files, <= 420 MB):

**Config A: 3 YOLO + EffNet-B3 bundle (SMALLEST CLASSIFIER)**
| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + EffNet-B3) | ~129 MB |
| **Total** | **346 MB** -- fits with 74 MB headroom |

**Config B: 3 YOLO + EffNetV2-S bundle**
| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + EffNetV2-S) | ~165 MB |
| **Total** | **382 MB** -- fits with 38 MB headroom |

**Config C: 3 YOLO + Swin-Tiny bundle**
| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + Swin-Tiny) | ~192 MB |
| **Total** | **409 MB** -- fits with 11 MB headroom |

**Config D: 2 YOLO + standalone classifier (NO BUNDLE NEEDED)**
| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| classifier_effnetv2_s_384.pt | 80 MB |
| **Total** | **297 MB** -- simple, no bundle complexity |

**Config E: Use existing bundle (yolov8m-640-v2s-bundle.pt)**
| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| yolov8m-640-v2s-bundle.pt | 155 MB |
| **Total** | **372 MB** -- fits, uses m-640 instead of l-640 |

### Recommendation: Config B or C for max accuracy, Config A for safety

Config B (EffNetV2-S at 384px, ~91% acc) is the sweet spot: high accuracy,
fits comfortably, 384px crops read grocery text well.

**But**: The tomorrow_3_shots.md plan says to use the existing
`yolov8m-640-v2s-bundle.pt` (Config E). This already exists and avoids creating
a new bundle, reducing risk of packaging errors. The tradeoff is m-640 (50MB)
vs l-640 (84MB) as the third YOLO model.

---

## 4. "BALLS OUT" IDEAS ANALYZED

### 4A. ONNX export for faster inference -> re-enable TTA with classifier
**Verdict: NOT WORTH IT.** We have 180s headroom at 127s. Even with classifier
adding ~20-30s, we are at ~160s. No need for ONNX optimization. The risk of
ONNX conversion bugs exceeds the timing benefit.

### 4B. Different WBF weights (not equal)
**Verdict: WORTH TRYING IN SUB 2 OR 3.**
Currently `weights=None` (equal weights). The x-1280 model is likely the strongest
single detector. Giving it higher weight could improve fusion quality.

Suggested: `weights=[1.0, 1.5, 0.8]` for [l-1280, x-1280, l-640]
or `weights=[1.0, 2.0, 1.0]`.

The ensemble_boxes WBF `weights` parameter controls how much each model's boxes
contribute to the fused box position and score. Higher weight = more influence.

**Implementation:** Add `WBF_MODEL_WEIGHTS` to constants.py, pass to
`weighted_boxes_fusion()` instead of `weights=None`.

### 4C. Per-model confidence weighting
**Verdict: LOW VALUE.** WBF already handles score averaging. Adding per-model
confidence scaling is redundant with WBF weights.

### 4D. Larger YOLO ensemble (4 models in 3 files via bundle)
**Verdict: INTERESTING BUT RISKY.**
Could bundle yolov8m-1280-corrected (50MB) with the classifier inside the l-640
bundle. Total: 85+132+(85+50+44) = 396 MB for 4 YOLO + EffNet-B3.
But: 4 models with TTA at 1280+1280+640+1280 resolution might exceed 300s.
And loading 4 models simultaneously needs ~400MB VRAM (should fit on L4's 24GB).
**Too risky for 3 remaining submissions.** Save a submission for this only if Sub 1
and Sub 2 both score below 0.915.

### 4E. nc=357 instead of 356
**Verdict: NON-ISSUE.** I verified: data.yaml has nc=356, IDs 0-355,
with 355=unknown_product. CLAUDE.md confirms "IDs 0-355; ID 355 = unknown_product".
The comment in data.yaml about "category 356 = unknown_product is inference-only"
is confusing but refers to the competition evaluation possibly including a 357th
category. Our models are trained with nc=356 and that is correct. No change needed.

### 4F. Post-processing we have not tried

**WBF conf_type parameter:**
`ensemble_boxes.weighted_boxes_fusion` has a `conf_type` parameter:
- `avg` (default): average scores across models
- `max`: take max score
- `box_and_model_avg`: more sophisticated averaging

Try `conf_type='max'` -- this preserves the highest confidence from any model,
which may improve detection mAP by keeping high-confidence true positives
from being diluted by models that missed the detection.

**Lower CONFIDENCE_THRESHOLD even further:**
Currently 0.005. Try 0.001 or even 0.0001. For mAP evaluation, more detections
at very low confidence extend the precision-recall curve. The cost is more
predictions to classify (slightly slower) but we have timing headroom.

**NMS IoU tuning:**
Currently IOU_THRESHOLD=0.50. Dense shelves have many adjacent products.
Try 0.55 or 0.60 to keep more overlapping boxes. This directly improves
recall in dense scenes.

### 4G. Classifier confidence gate tuning
**This is the single highest-leverage knob after enabling the classifier.**
- Gate too high (0.90): classifier barely overrides, minimal classification gain
- Gate too low (0.10): classifier overrides everything, including cases where
  YOLO was right and classifier is wrong
- Sweet spot depends on relative accuracy: if classifier is much better than YOLO,
  lower gate is better

We have no offline eval. The gate must be tuned across submissions.
- Sub 1: gate=0.70 (conservative)
- Sub 2: adjust based on Sub 1 results

---

## 5. OVERNIGHT TRAINING JOBS

### Should we launch anything RIGHT NOW?

**NO.** Here is why:
1. Any training job takes 2-4 hours minimum (dataset download + training + upload)
2. We have 6 classifiers already trained, all ~90-91% accuracy
3. The problem is NOT classifier quality -- it is integration bugs (now fixed)
4. A new YOLO model takes 8-12 hours to train. Results would arrive mid-Sunday
   with no time to test
5. The marginal gain from a slightly better classifier (92% vs 91%) is tiny
   compared to getting the existing classifier working correctly

**The right move is to focus all energy on the 3 submissions, not on training.**

One exception: if we discover a fundamental issue (e.g., classifier trained on
wrong label mapping), then we would need emergency retraining. But this is unlikely.

---

## 6. THE EXACT PLAN FOR TOMORROW'S 3 SUBMISSIONS

### Pre-submission setup (TONIGHT, before sleep)

1. **Create the bundle file** (if using Config B/C/E):
   - Verify the existing `yolov8m-640-v2s-bundle.pt` loads correctly
   - OR create a new bundle with l-640 + EffNetV2-S
   - Test that run.py starts without crashing

2. **Prepare two configs** in constants.py (save as separate files):
   - `configs/sub1_classifier.py` - classifier enabled
   - `configs/sub2_fallback.py` - no classifier, threshold tuning

3. **Verify submission packaging** works with `create_submission.py`

4. **Delete junk from weights/**: `_tmp_yolo.pt`, `model.pt`

### SUBMISSION 1 (Sunday morning, ~08:00): Fixed Classifier

**THE MOST IMPORTANT SUBMISSION. This tests whether the classifier helps AT ALL.**

```python
# constants.py changes:
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8m-640-v2s-bundle.pt",  # bundle: m-640 + V2S classifier
]
ENSEMBLE_IMAGE_SIZES = [1280, 1280, 640]

BUNDLE_WEIGHT_PATH = "weights/yolov8m-640-v2s-bundle.pt"

USE_CLASSIFIER = True
CLASSIFIER_MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"  # verify exact timm name
CLASSIFIER_INPUT_SIZE = 384
CLASSIFIER_CONFIDENCE_GATE = 0.70
SCORE_FUSION_ALPHA = 1.0         # DO NOT TOUCH -- no score fusion
USE_CLASSIFIER_TTA = False       # DO NOT TOUCH -- no h-flip

USE_TTA = True                   # YOLO TTA stays on
CONFIDENCE_THRESHOLD = 0.005
IOU_THRESHOLD = 0.50
WBF_IOU_THRESHOLD = 0.50
WBF_SKIP_BOX_THRESHOLD = 0.001

USE_PROTOTYPE_MATCHING = False   # Too risky without offline eval
```

**Why m-640 instead of l-640 for the bundle?**
The existing `yolov8m-640-v2s-bundle.pt` is pre-built and tested (155MB).
Creating a new l-640 bundle risks packaging errors. The m-640 is slightly
weaker than l-640 but still contributes meaningfully to the ensemble.
Total: 85 + 132 + 155 = 372 MB. Fits.

**Alternative if bundle approach is risky:** Use Config D (2 YOLO + standalone
classifier). Simpler, no bundle extraction needed. But loses the 3rd YOLO model.
Given that dropping from 3 to 2 YOLO previously cost 0.015+ in score, the bundle
approach is worth the complexity.

**Expected score: 0.915 to 0.925** (if classifier works as expected)

**Decision point:** Wait for Sub 1 results before configuring Sub 2.

---

### SUBMISSION 2 (Sunday ~11:00): Conditional on Sub 1 Results

**SCENARIO A: Sub 1 > 0.920 (classifier clearly helping)**
-> Push harder on classification:
- Lower CLASSIFIER_CONFIDENCE_GATE to 0.50 (override more detections)
- Try WBF weights=[1.0, 1.5, 0.8] to boost x-1280 contribution
- Lower CONFIDENCE_THRESHOLD to 0.001
- Raise IOU_THRESHOLD to 0.55 (keep more boxes in dense scenes)

**SCENARIO B: Sub 1 between 0.910 and 0.920 (classifier helping a little)**
-> Fine-tune the gate:
- Try CLASSIFIER_CONFIDENCE_GATE = 0.50 (more aggressive)
- Or try a different classifier (swap V2S for EffNet-B3 at 44MB -- create new bundle)
- Try WBF conf_type='max' if implementable

**SCENARIO C: Sub 1 < 0.910 (classifier hurting or neutral)**
-> Abandon classifier, go pure detection optimization:
- USE_CLASSIFIER = False
- Revert to proven 3 YOLO ensemble (l-1280, x-1280, l-640)
- CONFIDENCE_THRESHOLD = 0.001
- IOU_THRESHOLD = 0.55
- WBF_IOU_THRESHOLD = 0.55
- WBF_SKIP_BOX_THRESHOLD = 0.0005
- This is a pure threshold sweep aiming for +0.003-0.005 over Sub 4

**SCENARIO D: Sub 1 crashed (exit code 1)**
-> Fix the crash, submit the same config again (this IS Sub 2)
- Most likely causes: bundle extraction failure, missing prototype_matcher.py,
  or wrong classifier model name
- Test locally first if possible

---

### SUBMISSION 3 (Sunday ~13:00, FINAL): Lock In Best Score

**This is our last shot. Strategy depends on Sub 1 and Sub 2 results.**

**If we have a score >= 0.920 from Sub 1 or Sub 2:**
- Resubmit that exact config unchanged (safety submission)
- Or make ONE small change: adjust confidence gate by +/-0.1 from best

**If our best is still 0.9095 (Sub 4):**
- Submit the pure detection optimization (Scenario C above if not already tried)
- Or try the "everything different" approach:
  - Swap ensemble: l-1280 + x-1280 + m-1280 (all at 1280px, might improve WBF fusion
    since all models see the same resolution)
  - ENSEMBLE_IMAGE_SIZES = [1280, 1280, 1280]
  - This tests whether resolution diversity (1280+1280+640) or model diversity
    (l+x+m all at 1280) is better for WBF

**If we have a score between 0.910 and 0.920:**
- Iterate on the best config with one change:
  - Lower/raise gate
  - OR add WBF model weights
  - OR change NMS threshold
  - Pick the change most likely to push us over 0.920

---

## 7. WHAT THE TOP TEAMS ARE LIKELY DOING

The top 4 teams (0.921-0.926) with 5-14 submissions have converged on very
similar scores. This suggests a common architecture:

1. **Multi-model YOLO ensemble with WBF** (same as us)
2. **Two-stage classifier** (this is the key difference -- they have it working)
3. **Tuned WBF parameters** on validation set
4. **Possibly**: ONNX/TensorRT optimization for speed, allowing more models
5. **Possibly**: classifier ensemble (2-3 classifiers averaged)
6. **Possibly**: per-class confidence calibration

The 0.016 gap is NOT about having fundamentally different approaches. It is about
execution quality: getting the classifier to work correctly, tuning thresholds,
and avoiding integration bugs. This is exactly what we are about to fix.

---

## 8. RISK MATRIX

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Bundle extraction crashes | 15% | HIGH | Test locally tonight; fallback to Config D (2 YOLO) |
| Classifier model name wrong in bundle | 10% | HIGH | Verify exact timm model name from training logs |
| Classifier hurts score | 20% | MEDIUM | SCORE_FUSION_ALPHA=1.0 caps downside; Sub 2 reverts if needed |
| Timing exceeds 300s | 5% | CRITICAL | 127s + ~30s classifier = ~160s; enormous headroom |
| prototype_matcher.py missing from zip | 10% | HIGH | Use create_submission.py (includes it); verify zip contents |
| All 3 subs crash | 5% | CATASTROPHIC | Our Sub 4 (0.9095) remains as best; test Sub 1 config locally |

---

## 9. PRE-FLIGHT CHECKLIST (Run before EVERY submission)

```
[ ] constants.py matches intended config (print all values)
[ ] USE_CLASSIFIER matches intended state
[ ] SCORE_FUSION_ALPHA = 1.0 (NEVER set below 1.0)
[ ] Weight files exist and total <= 420 MB
[ ] Exactly 3 weight files in submission (or fewer)
[ ] run.py has no import os, subprocess, socket, etc.
[ ] src/prototype_matcher.py included in zip
[ ] python run.py --input test_images/ --output /tmp/test.json runs without crash
[ ] Output JSON is valid and has correct format
[ ] Total .py files in zip <= 10
[ ] No _tmp_yolo.pt or model.pt in weights/
```

---

## 10. SUMMARY: THE PATH TO TOP 10

| Step | Action | Expected Score |
|------|--------|---------------|
| Current | Sub 4: 3 YOLO + WBF + TTA, no classifier | 0.9095 |
| Sub 1 | + V2S classifier (all bugs fixed, gate=0.70, alpha=1.0) | **0.917 - 0.925** |
| Sub 2 | Tune gate/thresholds based on Sub 1 feedback | **0.920 - 0.928** |
| Sub 3 | Lock in best OR try alternative approach | **best of all** |

**The classifier is the missing piece. The bugs are fixed. The math says it should
work. Sub 1 tomorrow morning is the moment of truth.**

If we hit 0.920+, we are in the top 10. If we hit 0.925+, we are on the podium.

---

## APPENDIX A: EXACT TIMM MODEL NAMES

Verify these before submission -- wrong name = crash:

```python
# EfficientNetV2-S (our V2S bundle)
"tf_efficientnetv2_s.in21k_ft_in1k"

# EfficientNet-B3
"efficientnet_b3"

# Swin-Tiny
"swin_tiny_patch4_window7_224"

# ConvNeXt-Small
"convnext_small.fb_in22k_ft_in1k"
```

Check the bundle file to confirm:
```python
import torch
bundle = torch.load("weights/yolov8m-640-v2s-bundle.pt", map_location="cpu")
print(bundle.get("classifier_model_name"))  # Must match exactly
```

## APPENDIX B: WBF WEIGHTS IMPLEMENTATION

If using non-equal WBF weights in Sub 2+:

In `constants.py`:
```python
WBF_MODEL_WEIGHTS: list[float] = [1.0, 1.5, 0.8]  # [l-1280, x-1280, m/l-640]
```

In `run.py` `run_ensemble_inference()`, change line 432:
```python
# FROM:
weights=None,
# TO:
weights=WBF_MODEL_WEIGHTS if WBF_MODEL_WEIGHTS else None,
```

## APPENDIX C: CONFIDENCE THRESHOLD SENSITIVITY

For mAP evaluation, lower thresholds are strictly better (more points on P-R curve)
UNLESS they cause timeout (more predictions = more classifier crops = slower).

Estimate: at CONFIDENCE_THRESHOLD=0.001 with 3 YOLO + TTA, we might get
~2000 detections per image instead of ~1000. With classifier, that doubles
the crop processing time: ~60s total instead of ~30s. Still well within budget.

---

*Written by the Professor. Saturday March 21, 2026, evening. Good luck tomorrow.*
