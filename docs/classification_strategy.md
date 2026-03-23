# Classification Strategy - Path to 0.85+ mAP

**Date:** 2026-03-20 (2 days to deadline)
**Current score:** 0.7084 (rank 95/157)
**Top score:** 0.9199
**Scoring:** 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

## 1. Root Cause Analysis: Why Classification is Near Zero

### The core problem: YOLO is trying to classify but failing badly

Our YOLO models are trained with `nc: 356` -- they output class predictions. But the
classification mAP is near zero (~0.01 estimated). Here is why:

**A. Extreme class imbalance destroys classification accuracy.**
- 355 unique classes appear in training, but distribution is wildly skewed
- Top class has 390 annotations; 96 classes have <= 10 samples; 32 classes have <= 5 samples
- Only 210 real shelf images in training; 1,577 reference images (single product, clean background)
- YOLO sees ~20,959 shelf annotations + 1,577 ref annotations = ~22,536 total
- With 356 classes, average is ~63 per class, but the long tail is brutal

**B. Similar-looking products are indistinguishable at YOLO resolution.**
- The 356 products are Norwegian grocery items -- many are nearly identical packages
  (e.g., multiple WASA knekkebrods, multiple EVERGOOD coffees, multiple egg varieties)
- YOLO at 640px resolution cannot read product labels -- classification requires
  recognizing fine text/logo differences
- Even at 1280px, YOLO crops are typically 50-150px per product on a shelf

**C. The two-stage classifier (EfficientNet-B3) is not yet deployed.**
- `weights/classifier.pt` does not exist -- the classifier has not finished training
- Even when trained, INPUT_SIZE=224 is too small; constants.py says 300 but
  train_classifier.py uses 224 -- these are inconsistent
- The classifier trains on only ~22K crops total across 356 classes -- not enough

**D. WBF ensemble averages class labels, compounding errors.**
- When 3 models disagree on class (which they do for 356 classes), WBF picks the
  majority label -- but with near-random classification, this is still random

### Score breakdown estimate
| Component | Weight | Our estimate | Max possible | Gap |
|-----------|--------|-------------|-------------|-----|
| Detection mAP@0.5 | 0.70 | ~0.70 (= 0.49) | ~0.95 (= 0.665) | 0.175 |
| Classification mAP@0.5 | 0.30 | ~0.01 (= 0.003) | ~0.85 (= 0.255) | 0.252 |
| **Total** | | **~0.493** | **~0.920** | **0.427** |

The classification gap (0.252 potential points) is much larger than detection (0.175).
**Fixing classification is the highest-ROI move.**

## 2. Asset Inventory

| Asset | Status | Notes |
|-------|--------|-------|
| 3x YOLO ensemble (L-1280, L-640, M-640) | Deployed, 219MB | Good detection, bad classification |
| YOLOv8x-640 weights | Available (132MB) | Not in ensemble (size limit) |
| Reference images | 329 products, ~6 angles each | Clean, product-only, barcode-indexed |
| metadata.json | Maps 329 products to class IDs | 27 products have no reference images |
| EfficientNet-B3 classifier | Training in progress | Not yet deployed |
| Vertex AI A100 | Unlimited credits | Can launch multiple jobs |
| Local 3080 (10GB VRAM) | Available | Good for testing, small training |
| 4 submissions remaining today | | Use wisely |
| Weight budget | 219MB used of 420MB | 201MB remaining for classifier |

## 3. Immediate Actions (Priority Order)

### ACTION 1: Fix the classifier training and get weights NOW [CRITICAL, 2h]

The EfficientNet-B3 classifier is our best hope. Issues to fix before deploying:

1. **INPUT_SIZE mismatch:** `train_classifier.py` uses `INPUT_SIZE = 224` but
   `constants.py` uses `CLASSIFIER_INPUT_SIZE = 300`. Fix train_classifier.py to use 300.
   EfficientNet-B3 native resolution is 300px -- using 224 loses fine-grained detail.

2. **Launch on Vertex AI immediately** with these parameters:
   ```
   --epochs 50 --batch-size 64 --lr 3e-4
   ```
   Use the existing `vertex-job-classifier.yaml` config.

3. **If the current training job is still running**, check its progress. If it has
   reasonable val accuracy (>30%), download those weights and submit immediately.

4. **Size budget:** EfficientNet-B3 state_dict is ~46MB. With 219MB ensemble, total
   is ~265MB -- well within 420MB limit.

### ACTION 2: Build a CLIP/similarity-based classifier as backup [HIGH, 3h]

The reference images are a gold mine we are barely using. Instead of training a
classifier from scratch, use them for zero-shot or few-shot matching:

**Approach: CLIP embedding similarity**
1. Pre-compute CLIP embeddings for all 329 product reference images (front view)
2. At inference time, crop each YOLO detection, compute its CLIP embedding
3. Find the nearest reference product by cosine similarity
4. Use the reference product's class ID

**Why this works:**
- CLIP is pre-trained on 400M image-text pairs -- it already understands product packaging
- No training needed -- just compute embeddings and save them (~5MB)
- Works even for the 96 classes with <10 training samples
- Reference images show the exact product appearance

**Concerns:**
- CLIP model size: ViT-B/32 is ~350MB -- too big for weight budget
- Solution: pre-compute reference embeddings offline, ship only the embeddings + a
  lightweight projection head (~10MB total)
- Or use `open_clip` ViT-B-16 with `torch.jit.trace` and quantize

**Alternative: Use reference images as a retrieval database with a lighter backbone.**
Pre-extract features with the EfficientNet-B3 we're already training. At inference,
compare each crop's features against stored reference features. This reuses a model
we're already shipping.

### ACTION 3: Retrain YOLO with classification-friendly augmentation [MEDIUM, 4h]

Launch a new YOLO training job on Vertex AI optimized for classification:

1. **Increase resolution to 1280 or 1600** -- classification needs to read labels
2. **Use copy-paste augmentation** -- `ultralytics` supports `copy_paste=0.5`
3. **Increase mosaic to 0** in last 10 epochs (close_mosaic) -- helps classification
4. **Add more aggressive augmentation** for rare classes
5. **Train YOLOv8x at 1280** -- larger model = better classification head

Command:
```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8x-1280-cls-optimized" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=YOUR_DOCKER_REGISTRY/trainer:latest \
  --command="python training/train.py --model yolov8x.pt --imgsz 1280 --batch 4 --epochs 100"
```

### ACTION 4: Use all YOLO models for classification voting [QUICK, 30min]

Instead of WBF (which averages class labels poorly), implement a smarter class
selection in the ensemble:

1. Keep WBF for bbox fusion
2. For the fused box's class label, use **majority voting** across all models
3. If no majority, use the prediction with highest confidence
4. Weight votes by model confidence

This is a code-only change in `run.py` -- no training needed.

### ACTION 5: Lower classifier confidence gate [QUICK, 5min]

`CLASSIFIER_CONFIDENCE_GATE = 0.5` is too conservative. With 356 classes, even a
good classifier will rarely exceed 50% softmax confidence. Change to 0.15-0.20.
The classifier is almost certainly better than YOLO at classification even at lower
confidence.

## 4. Training Jobs to Launch RIGHT NOW on GCP

| Priority | Job | GPU | Est. Time | Expected Gain |
|----------|-----|-----|-----------|---------------|
| 1 | EfficientNet-B3 classifier (fix INPUT_SIZE=300, 50 epochs) | A100 | 1-2h | +0.10-0.15 score |
| 2 | YOLOv8x at 1280px, 100 epochs | A100 | 4-6h | +0.02-0.05 score |
| 3 | ConvNeXt-Small classifier (better than EfficientNet for fine-grained) | A100 | 2-3h | +0.02-0.05 over EffNet |
| 4 | YOLOv8l at 1600px (if budget allows) | A100 | 6-8h | Marginal |

## 5. What to Do on the Local 3080 (10GB VRAM)

The 3080 is perfect for:

1. **Run `eval_offline.py`** on current weights to establish exact baseline numbers
2. **Test the classifier pipeline end-to-end** with dummy weights
3. **Pre-compute CLIP reference embeddings** (inference only, fits in 10GB)
4. **Quick training experiments** with a frozen backbone classifier on shelf crops
5. **Validate submission zip** before each upload
6. **Profile inference timing** to ensure we stay under 300s budget

## 6. Fastest Path to 0.85+ Score

### Phase 1: Quick wins (today, March 20)
- Fix and deploy EfficientNet-B3 classifier -> estimated score: 0.75-0.80
- Lower CLASSIFIER_CONFIDENCE_GATE to 0.15 -> +0.01-0.02
- Submit and measure actual improvement

### Phase 2: Optimization (March 21)
- Train ConvNeXt-Small or EfficientNet-V2 classifier with:
  - Mixup/CutMix augmentation
  - Focal loss instead of CrossEntropy (handles class imbalance)
  - INPUT_SIZE=384 (more detail)
  - Label smoothing 0.1 (already have this)
  - Weighted sampling for rare classes
- CLIP reference matching as fallback for low-confidence classifier predictions
- New YOLO model at higher resolution

### Phase 3: Final submissions (March 22)
- Best classifier + best YOLO ensemble
- Tune confidence gates and thresholds based on submission feedback
- Consider dropping TTA if timing is tight with classifier overhead

### Projected scores:
| Configuration | Detection | Classification | Score |
|--------------|-----------|----------------|-------|
| Current (no classifier) | 0.70 | 0.01 | 0.493 (submitted as 0.708?) |
| + EfficientNet-B3 (30% acc) | 0.70 | 0.25 | 0.565 |
| + Better classifier (50% acc) | 0.70 | 0.45 | 0.625 |
| + Better YOLO + classifier (60% acc) | 0.75 | 0.55 | 0.690 |
| + All optimizations | 0.80 | 0.70 | 0.770 |
| Top teams (estimate) | 0.90 | 0.90 | 0.900 |

Note: The submitted score of 0.7084 seems higher than 0.7*0.70 + 0.3*0.01 = 0.493.
This suggests either (a) YOLO classification is not as bad as assumed -- maybe
~0.35 classification mAP, or (b) the scoring formula works differently than expected.
**Run eval_offline.py immediately to get exact detection vs classification breakdown.**

## 7. Critical Code Fixes Needed

### Fix 1: INPUT_SIZE consistency
In `training/train_classifier.py` line 49: change `INPUT_SIZE = 224` to `INPUT_SIZE = 300`
to match `constants.py` `CLASSIFIER_INPUT_SIZE = 300`.

### Fix 2: CLASSIFIER_CONFIDENCE_GATE
In `src/constants.py` line 86: change from `0.5` to `0.15`.

### Fix 3: Add weighted sampling for rare classes
In `training/train_classifier.py`, add a `WeightedRandomSampler` to oversample
classes with few examples. Classes with 1-5 samples need 50-100x oversampling.

### Fix 4: Consider using reference images at inference time
The 329 products with multi-angle reference images are not used at inference at all.
A simple nearest-neighbor lookup against pre-computed embeddings could handle the
27 classes with no reference images by falling back to the classifier.

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Classifier not ready by deadline | HIGH | Use CLIP zero-shot as backup |
| Classifier too slow (300s budget) | MEDIUM | Batch crops, use half precision, limit to top-300 detections |
| Weight files exceed 420MB | MEDIUM | Quantize classifier to int8 (~12MB) |
| Classifier makes detection worse | LOW | Only override category_id, never bbox/score |
| New YOLO training doesn't converge in time | MEDIUM | Stick with current ensemble |

## 9. Summary: Top 3 Actions in Order

1. **RIGHT NOW:** Launch EfficientNet-B3 classifier training on Vertex AI with INPUT_SIZE=300, 50 epochs, weighted sampling. Also run `eval_offline.py` locally to get exact baseline split.

2. **Within 2 hours:** Download classifier weights (even partially trained), set CLASSIFIER_CONFIDENCE_GATE=0.15, test end-to-end, submit.

3. **By end of today:** Launch YOLOv8x-1280 training on Vertex AI. Start building CLIP reference embedding pipeline as classifier backup.
