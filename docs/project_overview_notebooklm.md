# NM i AI 2026 — NorgesGruppen Grocery Detection: The Full Story

## The Competition

**NM i AI** (Norwegian Championship in AI) 2026 challenges teams to build an AI system that detects and classifies grocery products on store shelves. Think of it as teaching a computer to look at a photo of a grocery store shelf and identify every single product — what it is, and exactly where it sits.

The dataset comes from **NorgesGruppen**, Norway's largest grocery retailer. Teams receive 248 shelf images containing 22,731 annotated product bounding boxes across **356 different grocery product categories** — everything from "FRØKRISP KNEKKEBRØD ØKOLOGISK 170G BERIT" to "EVERGOOD DARK ROAST HELE BØNNER 500G".

### Scoring

The competition uses a weighted mAP (mean Average Precision) score:

- **70% Detection mAP**: Did you find the products? A bounding box with IoU ≥ 0.5 counts as detected, regardless of category.
- **30% Classification mAP**: Did you identify them correctly? The bounding box must overlap AND have the correct category_id.

This means detection is worth more than twice as much as classification. But at the top of the leaderboard, detection is nearly saturated — classification becomes the differentiator.

### Constraints

- **300 seconds** to process the entire test set on an NVIDIA L4 GPU (24GB VRAM, 8GB RAM)
- **420 MB** maximum total weight file size
- **Maximum 3 weight files** in submission
- **No internet access** at runtime — everything must be bundled
- **Restricted Python imports** — no `os`, `subprocess`, `socket`, `threading`, etc.
- **Pre-installed packages only**: ultralytics 8.1.0, PyTorch 2.6.0, timm, ensemble-boxes
- **6 submissions per day** (3 per day, rolling)

---

## Our Journey: From 0.70 to 0.90 in One Day

### The Starting Point

We began with a basic YOLOv8m model trained on the competition dataset. Single model, default settings, confidence threshold 0.25. Our first submission scored **0.7084 mAP**, ranking **#95 out of 157 teams**.

The leader was at **0.9199 mAP**. We had a gap of 0.21 to close.

### The Breakthrough: Multi-Model Ensemble

Our second submission transformed the pipeline completely:

**Three YOLO models running in parallel:**
- YOLOv8l at 1280px resolution (85MB) — our strongest single model (mAP50 = 0.796)
- YOLOv8l at 640px resolution (85MB) — fast, good at medium-sized products
- YOLOv8m at 640px resolution (51MB) — architectural diversity with a medium backbone

**Weighted Box Fusion (WBF)** merges predictions from all three models. When multiple models detect the same product, WBF combines their bounding boxes into a single, more accurate prediction. This is far superior to simple NMS because it averages box coordinates rather than just picking one.

**Test-Time Augmentation (TTA)** runs each model on flipped/scaled variants of the input image and merges the results. This catches products that might be missed at a single scale.

**Ultra-low confidence threshold (0.01)** ensures we capture every possible detection. For mAP evaluation, recall matters — the precision-recall curve needs the full range of predictions.

Result: **0.9025 mAP**, jumping to **#34 out of 203 teams**. A single architectural change — from single model to ensemble — delivered a **+0.194 improvement**.

The leader was now at **0.9200 mAP**. Gap reduced to just **0.0175**.

---

## Architecture Deep Dive

### The Inference Pipeline

```
Input Images
    |
    v
[YOLO Model 1: YOLOv8l @ 1280px] --\
[YOLO Model 2: YOLOv8l @ 640px]  ---+-> Weighted Box Fusion (WBF)
[YOLO Model 3: YOLOv8m @ 640px]  --/         |
                                              v
                                    Fused Detections
                                              |
                                              v
                                    [Crop each detection]
                                              |
                                              v
                            [Classifier Ensemble (EfficientNet + ConvNeXt)]
                                    + Classifier TTA (horizontal flip)
                                    + Prototype Matching (reference images)
                                              |
                                              v
                                    [Score Fusion: blend YOLO + classifier confidence]
                                              |
                                              v
                                    Final Predictions (JSON)
```

### Detection Stage

Each YOLO model independently processes every input image. The three models provide diversity through:
- **Resolution diversity**: 1280px sees fine details (product labels), 640px captures the big picture
- **Architecture diversity**: YOLOv8l (large) has more capacity, YOLOv8m (medium) provides a different learned representation

WBF with IoU threshold 0.55 fuses overlapping boxes. This means if two or more models detect the same product, their bounding boxes are averaged into a single, more precise box. The confidence scores are also merged.

### Classification Stage: The Two-Stage Approach

YOLO's built-in classifier struggles with 356 fine-grained grocery categories. Many products look nearly identical — multiple variants of WASA knekkebrød, several EVERGOOD coffee types, numerous egg brands. At 640px or even 1280px, the product-level crops are often just 50-150 pixels, making text illegible.

Our solution: **crop each detection and re-classify with a dedicated classifier**.

**EfficientNet-B3** (and ConvNeXt-Small as an alternative) are trained specifically on product crops:
- Reference product images (1,577 clean studio photos, multi-angle)
- Shelf image crops (22,731 annotated bounding boxes)
- Input size: 300-384px (much higher effective resolution than YOLO's classification head)
- Weighted sampling to oversample rare classes (96 categories have ≤10 training examples)

**Classifier TTA**: Each crop is run through the classifier twice — original and horizontally flipped — and softmax outputs are averaged. This simple trick typically adds 1-3% accuracy.

**Classifier Ensemble**: When multiple classifiers are available (e.g., EfficientNet + ConvNeXt), their softmax outputs are averaged before taking the argmax. Different architectures capture different visual features, improving robustness.

### The Secret Weapon: Prototype Matching

We have 1,577 reference product images — clean, studio-quality photos of 329 products from multiple angles. These are **completely unused at inference** in the standard pipeline.

**Prototype matching** changes that:

1. **Offline**: Extract feature embeddings from each reference image using the classifier's penultimate layer. Average embeddings per product to create "prototype" vectors. Save as a ~2MB tensor file.

2. **At inference**: When the classifier has low confidence on a detection crop (softmax < 0.5), extract its features and compute cosine similarity against all prototypes. If the nearest prototype is similar enough (> 0.6), use the prototype's category instead.

This is especially powerful for the 96 rare classes where the classifier has seen very few training examples but we have clean reference photos of the exact product.

### Score Fusion

The final prediction score blends YOLO's detection confidence with the classifier's confidence:

```
final_score = 0.7 * yolo_confidence + 0.3 * classifier_confidence
```

This improves the precision-recall curve by promoting detections where both the detector and classifier agree.

---

## Training Infrastructure

### GPU Fleet on Google Cloud

We run all training on **Vertex AI** with **NVIDIA A100 GPUs** and unlimited credits. At peak, we had **9 concurrent training jobs**:

**YOLO Detector Training:**
- YOLOv8l at 1280px (our best architecture, mAP50 = 0.796)
- YOLOv8x at 1280px (larger model, mAP50 = 0.784)
- YOLOv8m at 1280px (compact ensemble member)
- All retrained on corrected labels after human review

**Classifier Training:**
- EfficientNet-B3 at 224px (v1, 88.7% val accuracy)
- EfficientNet-B3 at 300px with weighted sampling (v2, 88.7%)
- ConvNeXt-Small at 384px (88.0% after 3 epochs, still climbing)
- EfficientNet-B3 at 384px with focal loss
- EfficientNet-V2-Small at 384px
- Swin-Tiny at 384px

### Label Corrections

We built a **Label Monkey** tool — a browser-based UI for reviewing suspicious annotations one at a time. A human reviewer examined 273 flagged items (tiny boxes, rare categories) and made 88 corrections:
- 62 relabeled (wrong category → correct category)
- 14 deleted (not a product)
- 12 marked as unknown_product

These corrections were uploaded to GCS and used for all retraining jobs.

### Data Augmentation Strategy

**YOLO Training:**
- Mosaic augmentation (4 images combined)
- Random scaling, rotation, flipping
- Color jitter
- FP16 (half precision) training

**Classifier Training:**
- RandomResizedCrop (0.6-1.0 scale)
- Horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (±15°)
- Random perspective distortion
- Random erasing (20% probability)
- Label smoothing (0.1)
- Weighted random sampling for class balance
- Optional focal loss for hard example mining

---

## The Weight Budget Game

With a 420MB limit and max 3 weight files, every megabyte counts:

| Component | Size | Purpose |
|-----------|------|---------|
| YOLOv8l-1280 | 85 MB | Primary detector (best mAP) |
| YOLOv8l-640 | 85 MB | Multi-scale ensemble member |
| YOLOv8m-640 | 51 MB | Lightweight ensemble diversity |
| EfficientNet-B3 classifier | 43 MB | Two-stage classification |
| Prototypes tensor | ~2 MB | Reference image matching |
| **Total** | **~266 MB** | **Within 420 MB** |

Alternative: swap YOLOv8m-640 (51MB) for ConvNeXt-Small classifier (190MB) when it finishes training. Total: ~405MB — still fits!

---

## Technology Stack

- **Detection**: Ultralytics YOLOv8 (v8.1.0) — industry-standard real-time object detector
- **Classification**: timm (PyTorch Image Models) — EfficientNet-B3, ConvNeXt-Small, Swin-Tiny
- **Ensemble**: ensemble-boxes library — Weighted Box Fusion
- **Training**: Vertex AI Custom Training with A100 GPUs
- **Framework**: PyTorch 2.6.0 with CUDA 12.4
- **Inference**: FP16 (half precision) for 2x GPU throughput
- **CI/CD**: GitHub + ruff linting + pytest (60 tests)

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Training images | 248 shelf photos |
| Total annotations | 22,731 product bounding boxes |
| Product categories | 356 |
| Reference product images | 1,577 (329 products, multi-angle) |
| Label corrections applied | 88 |
| YOLO models trained | 9 |
| Classifier models trained | 6 |
| Concurrent GPU jobs | 9 (peak) |
| Test suite | 60 tests, 0 failures |
| Score improvement | 0.7084 → 0.9025 (+0.194, +28%) |
| Rank improvement | #95 → #34 (out of 203 teams) |
| Gap to #1 | 0.0175 (0.9025 vs 0.9200) |
| Lines of code | ~2,500 (inference + training + scripts) |
| Commits in one day | 10+ |
| Time from start to 0.90 | ~8 hours |

---

## The Race to #1

With a gap of just 0.0175 to the leader, we have multiple paths to close it:

1. **Deploy the classifier** — 88.7% accuracy EfficientNet-B3, expected +0.005-0.015
2. **Classifier ensemble** — average EfficientNet + ConvNeXt outputs, expected +0.005-0.015
3. **Prototype matching** — leverage 1,577 unused reference images, expected +0.005-0.020
4. **Corrected-label YOLO models** — 3 models retraining on cleaned data, expected +0.005-0.015
5. **Classifier TTA + score fusion** — already implemented, expected +0.003-0.008

Conservative total: +0.015-0.030. Optimistic: +0.035-0.060.

**Target: 0.920-0.935+**

The next 48 hours will determine if we can overtake the leader. The classifier is training. The code is ready. The fleet is running. We just need the weights to land.

---

## Team

- **Human**: Strategy, label corrections, submission management
- **Claude (Lead Agent)**: Architecture, implementation, monitoring, coordination
- **Strategist Agent**: Competition analysis, training plans, score optimization
- **QA Agent**: Testing, security audits, code quality
- **Model Agent**: Training job management, weight downloads
- **9 A100 GPUs**: The silent workhorses running 24/7 on Google Cloud
