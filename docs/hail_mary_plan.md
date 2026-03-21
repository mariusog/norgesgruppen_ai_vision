# Hail Mary Plan -- Saturday Night, March 21, 2026

**Current best: 0.9095 (Sub 4). Leader: 0.9255. Gap: 0.016. Deadline: tomorrow evening.**
**Submissions: 1 left today, 3 tomorrow.**

---

## Part 1: WHY the classifier hurts -- Root Cause Analysis

### Five compounding bugs, ranked by severity

#### BUG 1 (CRITICAL): Score fusion destroys detection mAP

Score formula: `final = 0.7 * det_mAP + 0.3 * cls_mAP`

Detection mAP is 70% of the score and depends entirely on **score ranking** (the precision-recall curve). Score fusion (`alpha=0.7`) corrupts this ranking:

```
YOLO score 0.95, classifier conf 0.30 -> fused 0.755  (dropped 0.20!)
YOLO score 0.95, classifier conf 0.91 -> fused 0.938  (dropped 0.012)
YOLO score 0.50, classifier conf 0.91 -> fused 0.623  (raised 0.12)
```

Score fusion compresses all scores toward the mean. A high-confidence true positive at 0.95 gets dragged down to 0.75 if the classifier is uncertain. This re-orders the P-R curve and **directly hurts the 70%-weighted detection mAP**, which is the dominant score component.

**This alone explains the entire score drop.** Even perfect classification cannot compensate if detection mAP drops by more than `(classification_gain * 0.3) / 0.7`.

#### BUG 2 (SEVERE): Resolution mismatch -- V2S trained at 384px, served at 300px

The bundle file `yolov8m-640-v2s-bundle.pt` contains:
- `classifier_model_name: "tf_efficientnetv2_s.in21k_ft_in1k"`
- Trained with `--input-size 384` (confirmed from training job args)
- NO `classifier_input_size` key stored in the bundle

At inference, `CLASSIFIER_INPUT_SIZE = 300` in `constants.py`. The code never reads input size from the bundle. So a model trained on 384x384 crops is fed 300x300 crops.

This is not just a resolution difference -- it is a **domain shift**. The model's BatchNorm running statistics, learned spatial attention patterns, and internal feature map scales were all calibrated for 384px. At 300px, accuracy degrades substantially -- likely from ~91% down to ~85% or worse.

#### BUG 3 (MODERATE): Confidence gate at 0.15 is effectively no gate at all

With 356 classes, uniform random softmax would give ~0.003 per class. Even a confused classifier will produce top-1 > 0.15 for nearly every input. This means the classifier overrides YOLO's category_id on essentially **100% of detections**, including the ones where YOLO was already correct and the classifier is wrong.

With ~9% error rate (91% accuracy on correct-resolution inputs, probably worse with the resolution mismatch), this means ~9% of all detections get their category_id **actively corrupted**.

#### BUG 4 (MINOR): Preprocessing mismatch between training and inference

Training val transforms:
```python
Resize(int(384 * 1.14))  # = Resize(438)
CenterCrop(384)
```

Inference transforms:
```python
crop.resize((300, 300), Image.BILINEAR)  # Direct resize, no center crop
```

Even if the input size were correct (384), the inference code does a direct `PIL.resize` instead of the `Resize(438) + CenterCrop(384)` pipeline used during training. This is a secondary domain shift that further degrades accuracy.

#### BUG 5 (MINOR): Classifier TTA (h-flip) may hurt for text-heavy products

Grocery products often have text labels as their distinguishing feature. Horizontal flip makes text unreadable. Averaging flipped and non-flipped softmax dilutes the correct signal from the non-flipped pass. However, the training data also used `RandomHorizontalFlip(p=0.5)`, so the model has seen flipped images. Impact is likely small but negative.

#### BUG 6 (MINOR): 10% box padding adds shelf/neighbor context

The 10% padding on each side means the crop includes neighboring products. For densely packed shelves, this adds confusing visual context from adjacent products. However, this is a minor contributor compared to bugs 1-3.

### Why Sub 9 (with TTA) scored WORSE than Sub 8 (without TTA)

Sub 8 (no TTA): 0.8939. Sub 9 (TTA): 0.8871.

TTA produces more detections (augmented variants get fused). More detections = more crops to classify. With the classifier actively hurting scores (bugs 1-3), more detections means more damage. TTA amplifies the classifier's negative impact.

---

## Part 2: Quantifying the damage

The score formula is `0.7 * det_mAP + 0.3 * cls_mAP`.

Sub 4 (no classifier): 0.9095. If we assume detection-only gives ~0.70 det_mAP at most, then:
- `0.7 * det + 0.3 * cls = 0.9095`
- YOLO alone provides both det and cls labels; the sub4 score reflects YOLO's classification quality.

Sub 9 (classifier): 0.8871.
- Drop = 0.0224. That is **2.24 percentage points** lost.
- For score fusion to be net positive, classification mAP improvement would need to exceed `(det_mAP_drop * 0.7) / 0.3`. Even a small det_mAP drop of 0.01 requires a cls_mAP improvement of 0.023 to break even.

The classifier's 91% accuracy sounds good, but YOLO's own classification head is already ~88-89% accurate on the test set (inferred from the 0.9095 score). The classifier's marginal improvement is at most 2-3%, but the score fusion penalty on detection mAP is much larger.

---

## Part 3: Today's submission -- RECOMMENDATION

### Option A: Safe -- Resubmit Sub 4 config with tuned thresholds (RECOMMENDED)
- `USE_CLASSIFIER = False`
- `USE_TTA = True`
- 3 YOLO models (l-1280, x-1280, m-640) with WBF
- Tune `CONFIDENCE_THRESHOLD` lower (try 0.005 or 0.001) to capture more detections for mAP
- Tune `WBF_IOU_THRESHOLD` (try 0.45 or 0.55)
- This is our proven 0.9095 config. Threshold tuning could squeeze out another 0.005-0.01.

### Option B: Aggressive -- Classifier with all bugs fixed
If you insist on trying the classifier, fix ALL of these:
1. **SCORE_FUSION_ALPHA = 1.0** (disable score fusion entirely -- keep YOLO scores)
2. **CLASSIFIER_CONFIDENCE_GATE = 0.70** (only override when classifier is very confident)
3. **CLASSIFIER_INPUT_SIZE = 384** (match training resolution)
4. **USE_CLASSIFIER_TTA = False** (disable h-flip averaging)
5. Keep the same 3 YOLO ensemble + TTA

With these fixes, the classifier would only override category_id on ~60% of detections (those where it is 70%+ confident) and would NEVER touch the score. Detection mAP would be identical to Sub 4. Classification mAP could improve by 1-2% on those 60% of detections.

Expected improvement: `0.3 * ~0.02 * 0.6 = ~0.004` -- maybe 0.0036 points.

### My strong recommendation for today's 1 submission:

**Go with Option A.** The classifier has too many interacting bugs and we have no way to validate offline. The safe play is threshold tuning on the proven Sub 4 config.

Specific constants to set:
```python
USE_CLASSIFIER = False
USE_TTA = True
CONFIDENCE_THRESHOLD = 0.005  # Lower than current 0.01
IOU_THRESHOLD = 0.50          # Keep current
WBF_IOU_THRESHOLD = 0.50      # Keep current
WBF_SKIP_BOX_THRESHOLD = 0.0005  # Lower than current 0.001
```

---

## Part 4: Tomorrow's 3 submissions (Sunday, deadline day)

### Submission 1 (morning): Fixed classifier, Option B settings
Apply all 5 fixes from Option B above. This is the first real test of whether the classifier helps AT ALL when properly configured.

Key changes in `src/constants.py`:
```python
USE_CLASSIFIER = True
SCORE_FUSION_ALPHA = 1.0          # NO score fusion
CLASSIFIER_CONFIDENCE_GATE = 0.70  # High gate
CLASSIFIER_INPUT_SIZE = 384        # Match training
USE_CLASSIFIER_TTA = False         # No h-flip
```

### Submission 2 (afternoon): Best of {Sub 1 result, Sub 4 config} + NMS tuning
- If Sub 1 beats 0.9095, use classifier config + tune gate (try 0.60 or 0.80)
- If Sub 1 loses, go back to no-classifier + try `IOU_THRESHOLD = 0.55` and `CONFIDENCE_THRESHOLD = 0.001`

### Submission 3 (evening, final): Best known config, polish
- Take the best scoring config from all submissions
- Final threshold polish
- This is the "lock it in" submission

### Training jobs to launch NOW (results by tomorrow morning):

**Job 1: YOLO model with corrected labels, larger resolution**
We should NOT launch new YOLO training -- our 3 corrected models are our best assets and we cannot swap them out in time.

**Job 2: NO new classifier training needed**
We already have 5 classifier weights. The problem was never the classifier quality -- it was the integration bugs (score fusion, resolution mismatch, low gate). Fix the integration, not the model.

---

## Part 5: Alternative approaches for the gap to 0.9255

The gap is 0.016. That is substantial. Realistic options:

### A. Better WBF weights (no training needed)
Currently `weights=None` (equal weights). If the x-1280 model is stronger than the others, giving it weight 2.0 vs 1.0 for the others could help. Try `weights=[1.0, 2.0, 0.5]` for [l-1280, x-1280, m-640].

### B. Post-processing: Soft-NMS instead of hard NMS
YOLO uses hard NMS internally. We could apply Soft-NMS after WBF to keep more overlapping boxes at reduced confidence. This can improve recall in dense shelf regions.

### C. Category-aware WBF
Currently WBF fuses based on IoU only, ignoring category. If two models predict different categories for the same box, WBF keeps the one with higher score. We could add logic to split by category before fusion, which would preserve minority-class detections.

### D. Score calibration instead of fusion
Instead of blending YOLO + classifier scores, use the classifier ONLY for category_id and keep YOLO scores completely untouched. Then apply Platt scaling or temperature scaling to adjust the score distribution for better mAP. But this requires a calibration set, which we do not have time for.

### E. The most promising untried idea: class-conditional confidence
For classes where YOLO is systematically confused (e.g., similar-looking products), use the classifier. For classes where YOLO is confident and accurate, skip the classifier entirely. This requires per-class accuracy analysis on the validation set, which we could do offline tonight.

---

## Part 6: Summary Action Items

### RIGHT NOW (next 30 minutes):
1. Set `USE_CLASSIFIER = False` in constants.py
2. Set `CONFIDENCE_THRESHOLD = 0.005`
3. Build and submit (Option A)

### TONIGHT (after submission):
4. Run offline eval with classifier fixes (Option B) on val set to validate improvement
5. Tune WBF weights on val set
6. Prepare 3 submission configs for tomorrow

### TOMORROW:
7. Submit fixed classifier (Sub 1)
8. Submit based on Sub 1 results (Sub 2)
9. Final submission with best config (Sub 3)

---

*Analysis by the Professor. Saturday March 21, 2026, ~5:00 PM.*
