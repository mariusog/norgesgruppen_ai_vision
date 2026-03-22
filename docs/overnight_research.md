# Overnight Research -- Sunday March 22, 2026 (Deadline Day)

**Current best:** 0.9095. **Leader:** 0.9255. **Gap:** 0.016. **Submissions left:** 3.
**Deadline:** 15:00 Oslo time today.

---

## FINDING 1 (CRITICAL): WBF conf_type='avg' is destroying single-model detections

**This is the single biggest untried improvement available to us.**

The current WBF call uses `conf_type='avg'` (the default). I tested the actual
`ensemble_boxes` library empirically and the results are stark:

| Scenario | conf_type='avg' | conf_type='max' |
|----------|----------------|-----------------|
| Box found by 1 of 3 models (score 0.90) | **0.300** | **0.900** |
| Box found by 2 of 3 models (0.90, 0.85) | **0.583** | **0.900** |
| Box found by all 3 (0.90, 0.85, 0.80) | **0.850** | **0.900** |

With `conf_type='avg'`, any detection that only one model finds gets its score
divided by the number of models. A true positive at 0.90 becomes 0.30. This
destroys the precision-recall curve ranking for detection mAP (70% of score).

Dense shelf scenes have many small/occluded products that only one model detects
(especially the 1280px models catching things the 640px model misses). All of
these get their scores crushed to ~0.30, ranking them below much weaker
consensus detections.

**`conf_type='max'` preserves the highest score from any model.** This is
strictly better for mAP evaluation because:
1. True positives found by any model keep their high rank
2. Consensus detections still get high scores (0.85 -> 0.90, modest boost)
3. False positives from one model also keep high scores, BUT these are rare
   because YOLO already does NMS internally

**Action:** Change one line in `run.py` line 432:
```python
# FROM:
fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=None, iou_thr=WBF_IOU_THRESHOLD,
    skip_box_thr=WBF_SKIP_BOX_THRESHOLD,
)
# TO:
fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=None, iou_thr=WBF_IOU_THRESHOLD,
    skip_box_thr=WBF_SKIP_BOX_THRESHOLD,
    conf_type='max',
)
```

**Expected impact:** +0.005 to +0.015 on detection mAP = +0.004 to +0.011 on
final score. This alone could close half the gap.

**Risk:** Low. Worst case, some false positives from single models get higher
rank, but the mAP calculation handles this via the precision-recall tradeoff.

**Alternative:** `conf_type='absent_model_aware_avg'` behaves identically to
`'avg'` for equal weights (both give 0.30 for single-model detections). Not
useful. `'box_and_model_avg'` also gives 0.30. Only `'max'` preserves scores.

---

## FINDING 2: WBF weights -- the l-1280 model should be weighted higher

Currently `weights=None` (equal weights [1.0, 1.0, 1.0]).

With `conf_type='max'`, unequal weights change the score formula:
- `score = accumulated / weights.max()`
- So a detection from the highest-weighted model gets score = original
- A detection from a lower-weighted model gets score = original * (its_weight / max_weight)

This means with `weights=[2.0, 1.5, 1.0]`, a detection only from model 2
(weight 1.0) gets score 0.90 * (1.0/2.0) = 0.45, while model 0 (weight 2.0)
gets 0.90.

**With conf_type='max', weights are less important** because max already
preserves individual model scores. The weight only matters for the denominator.

**With conf_type='avg', weights DO matter** but the penalty for single-model
detections is still severe regardless of weighting.

**Recommendation:** If using `conf_type='max'`, keep `weights=None` (equal).
The max operation already does what we want. Unequal weights with max would
penalize detections from lower-weighted models, which is counterproductive.

If sticking with `conf_type='avg'`, try `weights=[1.5, 1.5, 1.0]` for the
two 1280px models -- but this is strictly worse than switching to `max`.

---

## FINDING 3: nc=356 vs nc=357 -- minor issue, not actionable

The system prompt says "357 categories (IDs 0-356; ID 356 = unknown_product)"
but our CLAUDE.md says "356 categories (IDs 0-355; ID 355 = unknown_product)."

Our data.yaml has nc=356 with class 355 = unknown_product. The comment says
"category 356 = unknown_product is inference-only." The test file
`test_output_format.py` uses `category_id: 356` as a valid prediction.

**Analysis:**
- Our models output class IDs 0-355. They can NEVER predict category 356.
- If the test set has objects labeled category 356, we always get classification
  wrong on those -- but detection is unaffected (category ignored for det_mAP).
- Impact: `0.3 * (fraction of cat-356 objects in test set)`. Likely tiny --
  unknown_product is a catch-all for rare/novel items.
- We cannot fix this without retraining (nc=357), which takes 8+ hours.

**Verdict:** Not actionable with 13 hours left. Accept the small loss.

---

## FINDING 4: Post-processing tricks for grocery/retail shelf detection

### 4A. Soft-NMS after WBF
YOLO uses hard NMS internally. After WBF fusion, we could apply Soft-NMS to
keep more overlapping detections at reduced confidence. Dense shelves have many
adjacent same-category products that hard NMS might suppress.

**However:** YOLO's internal NMS already ran at IoU=0.50, so most hard
suppressions already happened. WBF then merges across models. Adding another
NMS pass has diminishing returns and adds code complexity.

**Verdict:** Not worth the risk with 3 submissions. The `conf_type='max'`
change is simpler and higher impact.

### 4B. Box coordinate refinement
WBF already produces weighted-average box coordinates. No further refinement
is feasible without a dedicated refinement model.

### 4C. Category-specific confidence calibration
For categories where all models tend to agree (easy products), the current
system works fine. For hard categories (similar-looking products), using the
classifier to re-rank would help. But this is what the classifier already does.

**Verdict:** Focus on getting the classifier working (Finding 5 below).

### 4D. Small object padding
For very small detections (< 32px), the 10% crop padding may be too small to
give the classifier enough context. Increasing padding to 20% for small objects
could help classification accuracy.

**Verdict:** Minor. Not worth a submission slot.

---

## FINDING 5: Classifier deployment -- the bugs are fixed, should we enable it?

All 4 bugs have been fixed in the code:
1. SCORE_FUSION_ALPHA = 1.0 (no score corruption)
2. CLASSIFIER_INPUT_SIZE = 384 (matches training)
3. CLASSIFIER_CONFIDENCE_GATE = 0.70 (high gate, conservative)
4. USE_CLASSIFIER_TTA = False (no h-flip)

**BUT USE_CLASSIFIER = False in constants.py.** The classifier is disabled.

The key question: can we trust the classifier in a blind submission?

**Arguments FOR enabling:**
- With SCORE_FUSION_ALPHA=1.0, detection mAP is IDENTICAL (score unchanged)
- Downside is capped: worst case, a few category overrides are wrong, costing
  0.3 * (error_rate_increase * override_fraction) in cls_mAP
- If classifier is 91% accurate and YOLO is ~80%, even with a 0.70 gate
  overriding ~60% of detections, expected gain is +0.02 cls_mAP = +0.006 score

**Arguments AGAINST:**
- Every previous classifier submission scored WORSE (but those had Bug 1-4)
- We cannot validate offline -- this is a blind bet
- The classifier adds ~20-30s to inference time (still within 300s budget)

**Risk analysis with current settings:**
- Gate at 0.70: only overrides when classifier is very confident
- Of those overridden, if classifier is 91% accurate, it gets ~91% right
- YOLO on those same detections is maybe ~80% accurate
- Net improvement on overridden subset: +11% * 0.60 (fraction overridden) = +6.6%
- This translates to roughly +0.02 cls_mAP = +0.006 score
- Maximum downside (if classifier is actually 70% accurate): -10% * 0.60 = -6%
  = -0.02 cls_mAP = -0.006 score

**The risk is roughly symmetric** around the gate at 0.70. The key bet is
whether the classifier is genuinely 91% accurate on the test distribution.

---

## FINDING 6: Confidence threshold and mAP evaluation

Current CONFIDENCE_THRESHOLD = 0.005. For COCO-style mAP, lower thresholds are
strictly better because they add more points to the precision-recall curve
without affecting the integral (low-confidence false positives simply extend
the curve to the right at low precision, which doesn't hurt the area).

**However:** with WBF `conf_type='avg'`, lowering the threshold dramatically
increases the number of very-low-score detections that get further crushed by
WBF averaging. With `conf_type='max'`, these keep their original scores.

Try lowering to 0.001. With `conf_type='max'`, this captures more marginal
detections while preserving their score rankings.

**Conversely:** if we raise the threshold to 0.01 or 0.02, we reduce the
number of detections and speed up inference, but lose recall. Not recommended.

---

## FINDING 7: Scoring formula optimization (0.7*det + 0.3*cls)

The gap decomposition (from final_plan.md):
- Us: det ~0.955, cls ~0.803 -> 0.9095
- Leader: det ~0.960, cls ~0.850 -> 0.9255
- Gap: ~0.005 det + ~0.047 cls

**The optimal strategy given 3 submissions:**

The detection mAP is already strong. The marginal return from detection
improvements is 0.7x the improvement. The marginal return from classification
improvements is 0.3x but there is MUCH more room to grow (+0.047 potential).

**Detection improvement via conf_type='max': +0.005-0.015 det = +0.004-0.011 score**
**Classification improvement via classifier: +0.02-0.04 cls = +0.006-0.012 score**
**Combined: +0.010-0.023 score -> 0.920-0.933**

This is exactly the range that would close the gap to the leader.

---

## THE 3-SUBMISSION PLAN

### Submission 1: conf_type='max' + no classifier (SAFEST BIG WIN)

Changes from current config:
```python
# In run.py, WBF call: add conf_type='max'
# In constants.py:
USE_CLASSIFIER = False      # Keep disabled (safe)
CONFIDENCE_THRESHOLD = 0.003  # Slightly lower
# Everything else unchanged
```

This tests the single highest-impact change (conf_type='max') in isolation.
If it works, we know the improvement is real and additive with the classifier.

**Expected: 0.915-0.922** (pure detection improvement)

### Submission 2: conf_type='max' + classifier enabled

If Sub 1 beats 0.9095 (proving conf_type='max' helps):
```python
# In run.py: keep conf_type='max'
# In constants.py:
USE_CLASSIFIER = True
CLASSIFIER_CONFIDENCE_GATE = 0.70
SCORE_FUSION_ALPHA = 1.0     # DO NOT CHANGE
USE_CLASSIFIER_TTA = False   # DO NOT CHANGE
CLASSIFIER_INPUT_SIZE = 384
```

Which classifier to use depends on what bundles exist. The
`yolov8l-640-v2s-bundle.pt` (V2S at 384px, 91% accuracy) is pre-built and fits:
85 + 132 + 216 = 433 MB. **DOES NOT FIT** (over 420MB).

The `yolov8l-640-swin-bundle.pt` (Swin-Tiny 224px, 91%): 85 + 132 + 244 = 461 MB.
**DOES NOT FIT.**

The `yolov8l-640-bundle.pt` (EffNet-B3 300px, 90%): 85 + 132 + 178 = 395 MB.
**FITS.** But note CLASSIFIER_INPUT_SIZE should be 300 for EffNet-B3 (or 384 per
the training -- CHECK which resolution this classifier was trained at).

**Alternative: standalone classifier (2 YOLO + classifier, no bundle):**
85 + 132 + 84 (V2S) = 301 MB. FITS. But loses the 3rd YOLO model.

**Best option for Sub 2:**
Use `yolov8l-640-bundle.pt` (395 MB, fits) with EffNet-B3. Set:
```python
ENSEMBLE_WEIGHTS = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
    "weights/yolov8l-640-bundle.pt",
]
BUNDLE_WEIGHT_PATH = "weights/yolov8l-640-bundle.pt"
CLASSIFIER_MODEL_NAME = "efficientnet_b3"
CLASSIFIER_INPUT_SIZE = 384  # Verify training resolution
```

**Expected: 0.920-0.930** (detection + classification improvements stacked)

### Submission 3: Best config from Sub 1-2, or alternative

If Sub 1 < 0.9095: revert to proven config (0.9095) and try only
`CONFIDENCE_THRESHOLD = 0.001` + `IOU_THRESHOLD = 0.55`.

If Sub 1 > 0.9095 but Sub 2 < Sub 1: use Sub 1 config (conf_type='max' alone).

If Sub 2 > Sub 1: try lowering CLASSIFIER_CONFIDENCE_GATE to 0.50 for more
aggressive classification override.

---

## WEIGHT BUDGET RECAP (for quick reference)

| Config | Files | Size | Fits? |
|--------|-------|------|-------|
| Current (3 YOLO, no clf) | l-1280 + x-1280 + l-640 | 302 MB | YES |
| + EffNet-B3 bundle | l-1280 + x-1280 + l-640-bundle | 395 MB | YES |
| + V2S bundle | l-1280 + x-1280 + l-640-v2s-bundle | 433 MB | NO |
| + Swin bundle | l-1280 + x-1280 + l-640-swin-bundle | 461 MB | NO |
| 2 YOLO + V2S standalone | l-1280 + x-1280 + clf_v2s | 301 MB | YES |

---

## SUMMARY: Priority-ordered action items

| Priority | Action | Expected gain | Risk |
|----------|--------|--------------|------|
| **P0** | Add `conf_type='max'` to WBF call | +0.004-0.011 | Very low |
| **P1** | Lower CONFIDENCE_THRESHOLD to 0.003 | +0.001-0.003 | Very low |
| **P2** | Enable classifier with EffNet-B3 bundle | +0.003-0.012 | Medium |
| **P3** | Tune CLASSIFIER_CONFIDENCE_GATE (0.50-0.80) | +0.001-0.003 | Low |
| Skip | WBF weights (unequal) | Negligible with max | N/A |
| Skip | Soft-NMS after WBF | +0.001 | Medium |
| Skip | nc=357 retraining | Cannot do in time | N/A |

**The conf_type='max' change is the biggest free lunch available.
It requires changing one line of code and has very low downside risk.**

---

*Research by the Professor. Sunday March 22, 2026, early morning.*
