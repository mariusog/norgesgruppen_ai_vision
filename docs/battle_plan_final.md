# Final Battle Plan: 0.9025 -> 0.9200+ in 48 Hours

**Date:** 2026-03-20 (evening)
**Deadline:** March 22 (Sunday evening)
**Current score:** 0.9025 mAP (rank #34/203)
**Leader:** 0.9200 mAP
**Gap:** 0.0175
**Submissions remaining:** 3 tonight + 6 Saturday + 3 Sunday = 12 total

---

## Score Decomposition and Gap Analysis

Score = 0.7 * det_mAP@0.5 + 0.3 * cls_mAP@0.5

Our 0.9025 likely decomposes as approximately:
- Detection mAP ~ 0.94 (contributes 0.658)
- Classification mAP ~ 0.815 (contributes 0.245)
- Total: 0.903

Leader at 0.9200 likely has:
- Detection mAP ~ 0.95 (contributes 0.665)
- Classification mAP ~ 0.85 (contributes 0.255)
- Total: 0.920

**Key insight:** At this level, the gap is tight. We need improvements on BOTH axes but classification has the higher marginal return: +0.03 cls_mAP = +0.009 score, whereas +0.03 det_mAP = +0.021 score. Actually detection improvement has higher weight, but classification has more room to improve.

**The 0.0175 gap can be closed by ANY combination of:**
- +0.025 detection mAP alone = +0.0175 score
- +0.058 classification mAP alone = +0.0175 score
- +0.015 det + +0.017 cls = +0.0175 score (most realistic)

---

## TIER 0: CRITICAL QUICK WINS (Do First, Hours 0-4)

### 0A. Deploy Best Classifier Immediately
**Expected gain:** +0.005-0.015 score
**Effort:** 1 hour
**Risk:** Low

The EfficientNet-B3 v1 at 88.7% val accuracy is ready or nearly ready. Even at this accuracy, it will significantly improve classification over raw YOLO labels.

**Actions:**
1. Download the best checkpoint from GCS:
   ```bash
   gsutil cp gs://ai-nm26osl-1792-nmiai/weights/classifier_efficientnet_b3.pt weights/classifier.pt
   ```
2. Verify `USE_CLASSIFIER = True` and `CLASSIFIER_CONFIDENCE_GATE = 0.15` in `src/constants.py`
3. The classifier overrides YOLO category_id only when its softmax confidence exceeds the gate. At 0.15 with 356 classes (random baseline = 0.28%), this is appropriately aggressive.
4. Test timing locally -- the classifier adds ~30-50ms per image for crop+classify of all detections.

**Why this matters:** Our current pipeline uses YOLO's built-in classification, which is mediocre for 356 fine-grained categories. A dedicated 88.7% accurate classifier on higher-resolution crops is a substantial upgrade.

### 0B. Tune Classifier Confidence Gate
**Expected gain:** +0.002-0.005 score
**Effort:** 30 minutes
**Risk:** Very low

Test gate values: {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}

The optimal gate depends on the relative accuracy of YOLO classification vs the standalone classifier. With 356 classes and YOLO achieving maybe 60-70% accuracy while the classifier hits ~89%, a very low gate (0.05-0.10) is likely optimal -- we want the classifier to override almost always.

**Implementation:** Change `CLASSIFIER_CONFIDENCE_GATE` in `src/constants.py`, run offline eval, compare scores.

### 0C. Add Classifier TTA (Test-Time Augmentation for Crops)
**Expected gain:** +0.003-0.008 score
**Effort:** 2 hours
**Risk:** Low-Medium (timing budget concern)

Standard competition trick for classifiers: run each crop through the classifier multiple times with augmentations and average the softmax outputs.

**Augmentations to use (fast, effective):**
1. Original crop
2. Horizontal flip
3. Small scale variation (resize to 110%, center crop)
4. Slight brightness/contrast jitter

This gives 4 forward passes per crop. With batch processing, the overhead is ~3-4x for classification but classification is already fast relative to YOLO detection.

**Implementation in `run.py` `classify_crops()`:**
```python
# For each crop, create augmented versions
# Stack all variants, run through classifier in one batch
# Average softmax outputs across augmentations
# Take argmax of averaged softmax
```

**Timing estimate:** If we have ~300 detections per image and 4 augmentations, that's 1200 crops at 64 batch size = 19 batches. At ~5ms per batch on L4, that's ~100ms per image. Across 50 test images, ~5 seconds total. Acceptable.

---

## TIER 1: HIGH-VALUE IMPROVEMENTS (Hours 4-16, Saturday Morning)

### 1A. Classifier Ensemble (Average Multiple Classifier Outputs)
**Expected gain:** +0.005-0.015 score
**Effort:** 3 hours
**Risk:** Low

By tomorrow morning, we should have 3-5 trained classifiers:
- EfficientNet-B3 v1 (224px) -- ~89% accuracy
- EfficientNet-B3 v2 (300px, weighted sampling) -- ~89%+
- ConvNeXt-Small (384px) -- expected ~90-92%
- Swin-Tiny (384px) -- expected ~89-91%
- EfficientNet-B3 (384px, focal loss) -- expected ~90%
- EfficientNetV2-S (384px) -- expected ~89-91%

**Ensemble strategy:**
1. **Simple softmax averaging** of top 2-3 classifiers. Load 2-3 best classifiers, run each on every crop, average their softmax probability vectors, take argmax. This is the most robust approach and consistently works in Kaggle competitions.
2. **Diversity matters more than individual accuracy.** Pick classifiers with different architectures (e.g., ConvNeXt + EfficientNet + Swin) rather than 3 EfficientNets.
3. **Weight budget:** Each classifier is ~46-83MB. With 3 YOLO models at 221MB, we have 199MB left. Can fit 2 classifiers (e.g., EfficientNet-B3 46MB + ConvNeXt-Small 83MB = 129MB, total 350MB).

**Implementation approach:**
```python
# In run.py, modify load_classifier() to load multiple classifiers
# In classify_crops(), run all classifiers, average softmax, take argmax
# The confidence gate applies to the averaged softmax
```

**Weight budget plan for best case:**
| Model | Size | Role |
|-------|------|------|
| YOLOv8l-1280 | 85MB | Primary detector |
| YOLOv8l-640 | 85MB | Ensemble diversity |
| YOLOv8m-640 | 51MB | Ensemble diversity |
| EfficientNet-B3 classifier | 46MB | Classifier 1 |
| ConvNeXt-Small classifier | 83MB | Classifier 2 |
| **Total** | **350MB** | **Within 420MB** |

If needed, drop YOLOv8m-640 (51MB) and add another classifier for better classification at the cost of slightly weaker detection ensemble.

### 1B. Reference Image Feature Matching (THE BIG UNTAPPED ASSET)
**Expected gain:** +0.005-0.020 score
**Effort:** 4-6 hours
**Risk:** Medium

We have 1,577 reference product images across 329 categories that are NOT USED AT INFERENCE. This is potentially the single most impactful unused asset.

**Approach: Prototype Network / Nearest-Centroid Classification**

The idea is devastatingly simple:
1. **Offline (pre-compute):** For each of the 329 products with reference images, extract feature embeddings using one of our trained classifiers (e.g., the penultimate layer of EfficientNet-B3). Average the embeddings across all angles for each product to get a "prototype" embedding. Save as a tensor file (~2MB).
2. **At inference:** For each YOLO detection crop, extract the same feature embedding. Compute cosine similarity against all 329 prototypes. Use the nearest prototype's category as the prediction.
3. **Hybrid:** Use the classifier's softmax output when confidence is high (>0.7), fall back to prototype matching when confidence is low. This is especially powerful for rare classes where the classifier has seen few training examples but we have clean reference images.

**Why this is powerful for this specific competition:**
- 96 classes have <=10 training samples. The classifier has very few examples to learn from.
- But many of these rare classes DO have reference images (clean, multi-angle, studio-quality).
- Feature matching uses NO additional weights -- just a precomputed tensor of centroids (~356 x 1536 floats = ~2MB).
- It provides a fundamentally different classification signal that complements softmax.

**Architecture:**
```python
# Extract features by removing the classifier head
feature_extractor = nn.Sequential(*list(classifier.children())[:-1])
# Or use timm's forward_features:
features = classifier.forward_features(crop_tensor)  # [B, C, H, W]
features = features.mean(dim=[2,3])  # Global average pool -> [B, C]
features = F.normalize(features, dim=1)  # L2 normalize

# Cosine similarity against precomputed prototypes
similarities = features @ prototypes.T  # [B, 329]
nearest_class = prototype_class_ids[similarities.argmax(dim=1)]
```

**Key detail:** The prototypes only cover 329 of 356 classes. For the remaining 27 classes, fall back to the softmax classifier. This is handled naturally -- if the nearest prototype distance is below a threshold, keep the classifier's prediction.

**Combination strategy (vote fusion):**
```python
# Score = alpha * classifier_softmax + (1-alpha) * prototype_similarity
# With alpha = 0.6 (trust classifier more when confident)
# Or: use classifier when confident, prototype matching otherwise
```

**Pre-computation script needed:** A small script that:
1. Loads the best classifier
2. Hooks into the penultimate layer
3. Runs all 1,577 reference images through it
4. Averages embeddings per class
5. Saves as `weights/prototypes.pt` (~2MB)

### 1C. Use Corrected-Label YOLO Models in Ensemble
**Expected gain:** +0.005-0.015 score
**Effort:** 1-2 hours (just download and swap)
**Risk:** Low

Three YOLO retraining jobs are running with corrected labels. When they complete:
1. Download all corrected weights
2. Evaluate each individually with `eval_offline.py`
3. Test in ensemble combinations
4. Keep the best 3-model combo within weight budget

**Expected improvement:** Label corrections fix 88 annotation errors. This should improve both detection AND classification from YOLO. Even a +0.01 detection mAP contributes +0.007 to final score.

### 1D. Optimize WBF Parameters for Corrected Models
**Expected gain:** +0.002-0.005 score
**Effort:** 1 hour
**Risk:** Low

When swapping ensemble members, re-sweep:
- `WBF_IOU_THRESHOLD`: test {0.45, 0.50, 0.55, 0.60, 0.65}
- `WBF_SKIP_BOX_THRESHOLD`: test {0.0005, 0.001, 0.005, 0.01}
- YOLO `IOU_THRESHOLD` (NMS): test {0.40, 0.45, 0.50, 0.55, 0.60}

Dense shelf images benefit from slightly higher NMS IoU (0.50-0.55) to avoid suppressing adjacent products.

---

## TIER 2: MEDIUM-VALUE IMPROVEMENTS (Hours 16-36, Saturday Afternoon/Evening)

### 2A. Category-Aware Post-Processing
**Expected gain:** +0.002-0.008 score
**Effort:** 3-4 hours
**Risk:** Medium

**Idea 1: Per-class confidence calibration.**
Different categories have different confidence distributions from YOLO. Rare classes tend to have lower confidence even when correct. Apply per-class scaling factors learned from the validation set.

**Implementation:**
1. Run inference on val set, collect (predicted_class, confidence, is_correct) tuples
2. For each class, find the optimal confidence threshold
3. At test time, multiply each prediction's score by a per-class weight

**Idea 2: Category-aware NMS.**
Products on shelves often appear in groups of the same category. Standard NMS might suppress valid duplicate detections. Conversely, it might keep two overlapping boxes with different (wrong) class labels.

After WBF fusion and classifier re-scoring:
1. If two nearby boxes (IoU > 0.3) have the SAME category, keep both (they're likely adjacent products of the same type)
2. If two overlapping boxes (IoU > 0.5) have DIFFERENT categories, keep only the higher-confidence one
3. This is essentially class-aware NMS, which is more appropriate for retail shelf scenes

**Idea 3: Score fusion.**
Combine the YOLO detection confidence with the classifier confidence:
```python
final_score = yolo_conf * alpha + classifier_conf * (1 - alpha)
```
This can improve the precision-recall curve by promoting correct classifications.

### 2B. Box Padding for Classifier Crops
**Expected gain:** +0.002-0.005 score
**Effort:** 1 hour
**Risk:** Very low

When cropping YOLO detections for the classifier, add 10-15% padding around the bounding box. This gives the classifier more context (shelf edge, neighboring products) which can help distinguish visually similar products.

**Implementation in `classify_crops()`:**
```python
# Add 10% padding
pad_w = bw * 0.10
pad_h = bh * 0.10
left = max(0, int(bx - pad_w))
upper = max(0, int(by - pad_h))
right = min(img_w, int(bx + bw + pad_w))
lower = min(img_h, int(by + bh + pad_h))
```

This is a trivially simple change that often gives meaningful accuracy gains for fine-grained classification because product labels are sometimes partially outside the tight bounding box.

### 2C. Soft-NMS Instead of Hard NMS
**Expected gain:** +0.001-0.005 score
**Effort:** 2 hours
**Risk:** Low

Soft-NMS (Bodla et al., 2017) decays the confidence of overlapping boxes rather than eliminating them. This is especially valuable for dense shelf scenes where products touch/overlap.

**Challenge:** Ultralytics YOLO uses hard NMS internally. We cannot easily change this. However, we CAN apply Soft-NMS as a post-processing step AFTER WBF fusion:

```python
# After WBF, apply Soft-NMS to the fused predictions
# torchvision.ops.nms doesn't support soft-NMS, but we can implement it:
# For each box in confidence-sorted order:
#   For all lower-confidence boxes with IoU > threshold:
#     Reduce their confidence by: score *= (1 - IoU)^sigma
```

Actually, since WBF already does a form of soft fusion, the gain here may be minimal. Lower priority.

### 2D. Multi-Scale Classifier Inference
**Expected gain:** +0.002-0.005 score
**Effort:** 2 hours
**Risk:** Low

Run the classifier at multiple crop resolutions and average:
- 224px (fast, global features)
- 300px (native EfficientNet-B3)
- 384px (more detail, reads text better)

Average softmax across scales. This captures both global shape and fine-grained text details.

**Trade-off:** 3x classifier inference time. Still fast relative to YOLO detection.

---

## TIER 3: SPECULATIVE HIGH-UPSIDE IDEAS (Hours 24-48, If Time Permits)

### 3A. Pseudo-Labeling on Test Set
**Expected gain:** +0.005-0.015 score
**Effort:** 4-6 hours
**Risk:** High (if not allowed by rules)

**Concept:** Use our current best model to predict on the test images. Take high-confidence predictions (>0.8) as pseudo-labels. Retrain the classifier on the combined training + pseudo-labeled test data. This is a standard semi-supervised technique in competitions.

**Why it works here:**
- The test set likely contains products from the same shelves/stores
- High-confidence predictions are usually correct
- The classifier gets more examples of each class, especially rare ones
- Does NOT require test labels, just our own predictions

**Execution:**
1. Run current best pipeline on test images
2. Extract crops with high-confidence category predictions
3. Add to classifier training set
4. Retrain classifier for 10-20 epochs (fine-tune, don't start from scratch)

**Risk assessment:** Check competition rules for any prohibition on using test images during training. If the evaluation server provides test images and we predict on them, using those predictions for self-training is a gray area in many competitions but is commonly used and usually allowed.

### 3B. Knowledge Distillation: Ensemble -> Single Strong Model
**Expected gain:** +0.003-0.008 score
**Effort:** 6-8 hours
**Risk:** Medium

If we're hitting timing limits with 3 YOLO models + 2 classifiers, distill the ensemble knowledge into a single strong YOLO model:
1. Run ensemble on training images to get soft labels
2. Train a new YOLOv8x on these soft labels at 1280px
3. Single model inference is 3x faster, freeing time for classifier ensemble + TTA

**Why this could work:** The single distilled model captures the "average wisdom" of the ensemble without the 3x inference cost. This frees timing budget for more classifier tricks.

**Skip unless timing is tight.** With L4 GPU and 300s budget, we probably have room for the full ensemble + classifier.

### 3C. Dynamic Confidence Thresholding Based on Image Complexity
**Expected gain:** +0.001-0.003 score
**Effort:** 3 hours
**Risk:** Medium

Some test images are dense shelves with 200+ products; others are sparse displays with 10 products. Use a simple heuristic (number of detections above medium confidence) to dynamically adjust the final confidence threshold:
- Dense image (many detections): raise threshold slightly to reduce false positives
- Sparse image: lower threshold for better recall

**Implementation:** After all detections are generated, count detections per image. If >200, apply a small confidence boost filter.

### 3D. Mixup/CutMix for Classifier Retraining
**Expected gain:** +0.002-0.005 score (via better classifier)
**Effort:** 2 hours to implement, 3-4 hours to train
**Risk:** Low

If retraining the classifier (for pseudo-labels or continued training), add:
- **Mixup** (alpha=0.2): linear interpolation of image pairs + label mixing
- **CutMix** (alpha=1.0): cut-paste regions between images + proportional label mixing

These are proven regularization techniques that improve fine-grained classification accuracy by 1-3%.

**Implementation:** Add to `train_classifier.py` training loop:
```python
# CutMix with 50% probability per batch
if random.random() > 0.5:
    lam = np.random.beta(1.0, 1.0)
    # ... standard CutMix implementation
```

---

## TIMING BUDGET ANALYSIS

**L4 GPU, 300 second budget, ~50 test images (estimate)**

| Component | Per-image | Total (50 imgs) | Notes |
|-----------|-----------|-----------------|-------|
| Image loading | ~5ms | 0.25s | |
| YOLO model 1 (L-1280, TTA) | ~200ms | 10s | |
| YOLO model 2 (L-640, TTA) | ~60ms | 3s | |
| YOLO model 3 (M-640, TTA) | ~40ms | 2s | |
| WBF fusion | ~5ms | 0.25s | |
| Classifier (300 crops, batch 64) | ~30ms | 1.5s | |
| Classifier 2 (300 crops) | ~30ms | 1.5s | |
| Classifier TTA (4x) | ~90ms | 4.5s | |
| Prototype matching | ~5ms | 0.25s | |
| Post-processing | ~2ms | 0.1s | |
| **Total** | **~467ms** | **~23s** | **WELL within budget** |

Even with 200 test images (unlikely), total would be ~93s. We have enormous headroom.

**If test set is larger (500 images):**
Total ~233s -- still within budget but tighter. Can drop classifier TTA if needed.

**Conclusion:** We can afford the full pipeline: 3 YOLO models + TTA + 2 classifiers + classifier TTA + prototype matching. No timing concerns unless test set exceeds ~600 images.

---

## WEIGHT BUDGET ANALYSIS

Maximum: 420 MB

**Optimal configuration:**
| File | Size | Purpose |
|------|------|---------|
| weights/yolov8l-1280-aug.pt (or corrected) | 85MB | Best detector |
| weights/yolov8l-640-aug.pt (or corrected) | 85MB | Multi-scale detector |
| weights/yolov8m-640-aug.pt (or corrected) | 51MB | Ensemble diversity |
| weights/classifier.pt (EfficientNet-B3) | ~46MB | Primary classifier |
| weights/classifier2.pt (ConvNeXt-Small) | ~83MB | Secondary classifier |
| weights/prototypes.pt | ~2MB | Reference feature centroids |
| **Total** | **~352MB** | **Within 420MB** |

**Alternative (swap m-640 for x-1280 corrected):**
| File | Size | Purpose |
|------|------|---------|
| weights/yolov8x-1280-aug.pt (corrected) | ~137MB | Strongest single detector |
| weights/yolov8l-640-aug.pt (corrected) | 85MB | Multi-scale |
| weights/classifier.pt (EfficientNet-B3) | ~46MB | Primary classifier |
| weights/classifier2.pt (ConvNeXt-Small) | ~83MB | Secondary classifier |
| weights/prototypes.pt | ~2MB | Reference feature centroids |
| **Total** | **~353MB** | **Within 420MB** |

This gives stronger single-model detection but fewer ensemble members. Test both configs.

---

## SUBMISSION STRATEGY

### Tonight (3 submissions)

**Sub 1 (ASAP):** Deploy best available classifier + current YOLO ensemble.
- Just add `classifier.pt`, set `USE_CLASSIFIER=True`, `CLASSIFIER_CONFIDENCE_GATE=0.10`
- Purpose: establish new baseline with classifier

**Sub 2 (if Sub 1 gains > +0.005):** Same config + classifier TTA + lower gate (0.05).
- Purpose: test if classifier TTA helps

**Sub 3:** HOLD for tomorrow unless a corrected-label YOLO model finishes tonight.

### Saturday (6 submissions)

**Sub 4:** Best corrected-label YOLO models + best single classifier
- Swap in corrected models, keep best classifier from tonight

**Sub 5:** Add classifier ensemble (2 classifiers)
- Download ConvNeXt-Small or Swin-Tiny classifier when ready
- Average softmax from both

**Sub 6:** Add prototype matching
- Pre-compute prototypes, add hybrid classifier+prototype scoring

**Sub 7:** Tune WBF + NMS thresholds for the new ensemble
- Re-sweep thresholds on val set with corrected models

**Sub 8:** Best overall config with all optimizations
- Combine best YOLO ensemble + best classifier ensemble + prototype matching

**Sub 9:** Alternative config (e.g., 2 YOLO + 2 classifiers + aggressive prototype matching)

### Sunday (3 submissions)

**Sub 10:** Fine-tuned version of best Saturday config
- Per-class confidence calibration applied
- Final threshold tuning

**Sub 11:** Safety submission -- resubmit best-scoring config unchanged
- In case of scoring server inconsistency

**Sub 12:** Hail Mary -- try the most aggressive config
- All tricks enabled, most classifiers, lowest thresholds

---

## HOUR-BY-HOUR EXECUTION PLAN

### Friday Evening (Hours 0-4)
- [x] Download best classifier checkpoint from GCS
- [ ] Deploy classifier in pipeline, test locally
- [ ] Submit Sub 1 (baseline + classifier)
- [ ] Evaluate classifier gate values on val set
- [ ] Monitor training jobs (corrected YOLO, additional classifiers)

### Saturday Morning (Hours 8-12)
- [ ] Download all overnight training results
- [ ] Evaluate all new YOLO models with `eval_offline.py`
- [ ] Evaluate all new classifiers
- [ ] Pick best YOLO 3-model ensemble
- [ ] Pick best 2 classifiers for ensemble
- [ ] Submit Sub 4 (new YOLO + classifier)

### Saturday Afternoon (Hours 12-20)
- [ ] Implement classifier TTA in `run.py`
- [ ] Implement classifier ensemble in `run.py`
- [ ] Implement prototype matching:
  - Pre-compute prototype embeddings script
  - Add prototype matching to `classify_crops()`
- [ ] Add box padding for classifier crops
- [ ] Submit Sub 5, Sub 6

### Saturday Evening (Hours 20-28)
- [ ] Sweep WBF/NMS thresholds with new models
- [ ] Implement category-aware post-processing
- [ ] Per-class confidence calibration
- [ ] Submit Sub 7, Sub 8, Sub 9
- [ ] If time: implement pseudo-labeling + classifier retrain

### Sunday (Hours 32-48)
- [ ] Final tuning based on Saturday submission results
- [ ] Submit Sub 10, 11, 12
- [ ] Ensure best-ever score is locked in

---

## WHAT A KAGGLE GRANDMASTER WOULD DO DIFFERENTLY

1. **Larger backbone models.** Top competitors in fine-grained product recognition use EVA-02, BEiT-3, or InternImage -- massive vision transformers pre-trained on billions of images. These are 300-600MB but have dramatically better feature representations. With our 420MB budget minus ~50MB for a single detector, we could fit one large classifier. However, these require `transformers` or `mmdet` which may not be pre-installed.

2. **Better training data augmentation.** Winners typically use:
   - **Copy-paste augmentation** (paste product crops onto random backgrounds)
   - **Mosaic-4 + mosaic-9** at training time
   - **Style transfer** to augment reference images with shelf-like backgrounds
   - **AugMax** adversarial augmentation

3. **Multi-task learning.** Train the classifier to jointly predict:
   - Category (356 classes)
   - Product brand/family (coarser grouping, ~50 groups)
   - Hierarchical classification with grouped softmax
   This leverages the natural hierarchy (all WASA products share visual features).

4. **Overlap between detection and classification optimization.** Top teams often:
   - Train the detector with frozen classification head, then unfreeze
   - Use a separate classification head at the YOLO feature map level
   - Apply deformable attention at the ROI level

5. **Test-time ensemble of augmented inputs.** Beyond YOLO TTA:
   - Run the entire pipeline on the original image AND a padded/scaled version
   - Merge results. This catches products at image edges.

6. **Leverage Norwegian text recognition.** The product categories ARE the product names. An OCR model (even lightweight) could read the product label text and match to known product names. This is essentially solving a text recognition problem disguised as an object detection problem. However, this requires an OCR model which may be too large/complex to deploy.

7. **Hard negative mining for the classifier.** After initial classifier training:
   - Run classifier on training set
   - Collect all misclassified examples
   - Retrain with emphasis on these hard negatives
   - This is essentially focal loss but applied at the data level

---

## RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Classifier makes score worse | Low | Medium | Test on val first; confidence gate prevents bad overrides |
| Corrected YOLO models not ready | Medium | High | Keep current ensemble as fallback |
| Weight budget exceeded | Low | Critical | Pre-calculate all sizes before submission |
| Timing exceeded on test set | Very Low | Critical | Profiled at ~23-93s for 50-200 images |
| Multiple classifiers slow inference | Low | Medium | Batch crops, profile, drop if needed |
| Prototype matching hurts accuracy | Low | Low | Only use for low-confidence classifier predictions |
| No improvement over 0.9025 | Medium | High | Submit current best as safety; try diverse strategies |

---

## SUMMARY: PRIORITY-RANKED ACTION LIST

| # | Action | Expected Gain | Time | Priority |
|---|--------|--------------|------|----------|
| 1 | Deploy classifier with low gate (0.10) | +0.005-0.015 | 1h | DO NOW |
| 2 | Swap in corrected-label YOLO models | +0.005-0.015 | 1h | When ready |
| 3 | Classifier ensemble (2-3 models) | +0.005-0.015 | 3h | Sat morning |
| 4 | Reference image prototype matching | +0.005-0.020 | 5h | Sat afternoon |
| 5 | Classifier TTA (4 augmentations) | +0.003-0.008 | 2h | Sat afternoon |
| 6 | Box padding for classifier crops | +0.002-0.005 | 30min | With #5 |
| 7 | Re-sweep WBF/NMS thresholds | +0.002-0.005 | 1h | After #2 |
| 8 | Category-aware post-processing | +0.002-0.008 | 3h | Sat evening |
| 9 | Per-class confidence calibration | +0.002-0.005 | 2h | Sat evening |
| 10 | Pseudo-labeling + classifier retrain | +0.005-0.015 | 5h | If time |
| 11 | Score fusion (YOLO conf + classifier conf) | +0.001-0.003 | 1h | Low priority |
| 12 | Soft-NMS post-processing | +0.001-0.005 | 2h | Low priority |

**Conservative total expected improvement:** +0.015-0.030
**Optimistic total:** +0.035-0.060
**Target: 0.920-0.935+**

The gap of 0.0175 is very closeable. Items 1-5 alone could deliver +0.020-0.050. The key is disciplined execution: deploy the classifier first, then layer improvements one at a time, validating each on the val set before spending a submission.

---

## CRITICAL REMINDERS

1. **Never `git add .`** -- stage specific files only
2. **No `import os`** in any `.py` file in the submission -- use `pathlib`
3. **Validate every submission** with `scripts/validate_submission.sh` before uploading
4. **Remove `model.pt`** from weights/ before creating submission zip (it's a duplicate)
5. **Run `eval_offline.py`** after every change to measure impact
6. **Profile timing** with `time python run.py` on representative images
7. **Keep the best-ever submission config saved** in a git branch as insurance
