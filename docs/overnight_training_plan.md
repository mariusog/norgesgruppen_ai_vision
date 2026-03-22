# Overnight Training Plan -- March 20-21, 2026

**Created:** 2026-03-20 evening
**Deadline:** March 22, 2026 (2 days away)
**Current score:** 0.7084 mAP (rank 95/157, top is 0.9199)
**Scoring:** 0.7 * detection_mAP + 0.3 * classification_mAP
**Submissions remaining:** 4 today, 3 tomorrow, 3 on deadline day
**Currently running:** Classifier v1 (224px, ~88.7% acc), v2 (300px, ~88.1% acc)

---

## Score Gap Analysis

Our 0.7084 implies roughly:
- Detection mAP ~0.80 (contributes 0.56)
- Classification mAP ~0.50 (contributes 0.15)

Top teams at 0.92 likely have:
- Detection mAP ~0.93 (contributes 0.65)
- Classification mAP ~0.90 (contributes 0.27)

**Biggest gains come from classification (+0.12 potential) AND detection (+0.09 potential).**

---

## Job Priority Tiers

### TIER 1: CRITICAL (launch immediately)

#### Job 1: Retrain YOLOv8l-1280 with corrected labels
**Rationale:** Our best model (mAP50=0.796) was trained on data with 88 label errors. Corrected labels should improve both detection AND classification from YOLO directly. This is the single highest-value job.

**Expected benefit:** +0.02-0.04 detection mAP (= +0.014-0.028 score)
**Risk:** Low -- same architecture that already works, just cleaner data
**Estimated time:** 4-5 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8l-1280-corrected-labels" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--data,training/data.yaml,--model,yolov8l.pt,--imgsz,1280,--batch,4,--epochs,150,--run-id,yolov8l-1280-corrected"
```

#### Job 2: Retrain YOLOv8x-1280 with corrected labels
**Rationale:** Second-best model (mAP50=0.784). With corrected labels, could surpass or complement YOLOv8l-1280 in an ensemble. x-model has more capacity for fine-grained classification.

**Expected benefit:** +0.02-0.04 detection mAP
**Risk:** Low
**Estimated time:** 6-8 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8x-1280-corrected-labels" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--data,training/data.yaml,--model,yolov8x.pt,--imgsz,1280,--batch,2,--epochs,150,--run-id,yolov8x-1280-corrected"
```

#### Job 3: ConvNeXt-Small classifier at 384px
**Rationale:** ConvNeXt consistently outperforms EfficientNet on fine-grained classification benchmarks. 384px gives more detail for reading product labels. This is the highest-impact classifier experiment.

**Expected benefit:** +0.03-0.08 classification mAP (= +0.009-0.024 score)
**Risk:** Medium -- new architecture, but timm has it pre-trained and it's well-proven
**Estimated time:** 2-3 hours on A100

**IMPORTANT:** The classifier upload always goes to `weights/classifier_efficientnet_b3.pt`. For ConvNeXt, we need to modify the GCS destination. The easiest approach: override GCS_WEIGHTS_NAME via the command line or modify train_classifier.py to accept a `--model-name` arg. Since we can't modify the container image easily, pass environment variables or modify the script to save to a run-id-based filename.

**Workaround:** The classifier script saves locally to `weights/classifier.pt` and uploads to `weights/classifier_efficientnet_b3.pt` on GCS. Since we want multiple classifiers to coexist on GCS, we should modify the script. However, since we can't rebuild the container quickly, we use a different approach: each job saves to the same GCS path, so we run them sequentially or download results between runs.

**Alternative:** Since we can't easily change GCS_WEIGHTS_NAME without rebuilding, launch this job AFTER the existing classifier jobs finish, or accept that it overwrites the GCS file (download previous results first).

For now, use the existing script with modified args. The `timm` library in the container (0.9.12) supports `convnext_small.fb_in22k_ft_in1k`:

```bash
# NOTE: This requires modifying train_classifier.py to accept --model-name and --input-size args.
# See "Required Code Changes" section below.
# After rebuilding the container:
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="convnext-small-384px-classifier" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--epochs,50,--batch-size,48,--lr,2e-4,--model-name,convnext_small.fb_in22k_ft_in1k,--input-size,384,--gcs-name,classifier_convnext_small_384.pt,--run-id,convnext-small-384"
```

---

### TIER 2: HIGH VALUE (launch after Tier 1)

#### Job 4: EfficientNet-B3 classifier at 384px with focal loss
**Rationale:** Even if ConvNeXt is better, having a second strong classifier enables classifier ensembling. 384px is larger than the current 300px runs. Focal loss handles class imbalance better than CrossEntropy for our long-tailed distribution (96 classes with <=10 samples).

**Expected benefit:** +0.02-0.04 classification mAP over current 300px
**Risk:** Low -- same architecture with better hyperparameters
**Estimated time:** 2-3 hours on A100

```bash
# Requires the --input-size, --loss, --gcs-name args added to train_classifier.py
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="effnet-b3-384px-focal" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--epochs,50,--batch-size,48,--lr,2e-4,--model-name,efficientnet_b3,--input-size,384,--use-focal-loss,--gcs-name,classifier_effnet_b3_384_focal.pt,--run-id,effnet-b3-384-focal"
```

#### Job 5: Swin-Tiny classifier at 384px
**Rationale:** Swin Transformers excel at fine-grained recognition tasks. Adding architectural diversity improves classifier ensembling. Swin-Tiny is ~28M params -- similar size to EfficientNet-B3 (~12M) but with attention-based feature extraction that may capture different product features.

**Expected benefit:** +0.02-0.05 classification mAP (diversity benefit)
**Risk:** Medium -- transformers can be slower at inference; need to verify timing
**Estimated time:** 3-4 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="swin-tiny-384px-classifier" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--epochs,50,--batch-size,48,--lr,2e-4,--model-name,swin_tiny_patch4_window7_224,--input-size,384,--gcs-name,classifier_swin_tiny_384.pt,--run-id,swin-tiny-384"
```

#### Job 6: Retrain YOLOv8m-1280 with corrected labels
**Rationale:** Smaller model for ensemble diversity. m-1280 trained on corrected data should outperform m-640 on corrupted data. Keeps the weight budget low (51MB) while benefiting from 1280px resolution.

**Expected benefit:** +0.01-0.02 detection mAP in ensemble
**Risk:** Low
**Estimated time:** 3-4 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8m-1280-corrected-labels" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--data,training/data.yaml,--model,yolov8m.pt,--imgsz,1280,--batch,8,--epochs,150,--run-id,yolov8m-1280-corrected"
```

---

### TIER 3: SPECULATIVE (launch if GPU slots available)

#### Job 7: YOLOv8l-1280 with copy-paste augmentation
**Rationale:** Copy-paste augmentation is the single most effective augmentation for dense object detection (shown in COCO benchmarks). It creates synthetic training examples by pasting object crops from one image onto another. Especially valuable for our rare classes. However, this requires passing `copy_paste=0.5` to `model.train()`, which our current train.py doesn't support as a CLI arg.

**Expected benefit:** +0.01-0.03 detection mAP
**Risk:** Medium -- requires code change to train.py to pass through extra ultralytics kwargs
**Estimated time:** 5-6 hours on A100

**Requires adding `--copy-paste` flag to train.py** (see code changes below), then rebuild container.

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8l-1280-copypaste" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--data,training/data.yaml,--model,yolov8l.pt,--imgsz,1280,--batch,4,--epochs,150,--run-id,yolov8l-1280-copypaste,--copy-paste,0.5"
```

#### Job 8: EfficientNet-V2-Small classifier at 384px
**Rationale:** EfficientNet-V2 is the successor to EfficientNet with improved training speed and accuracy. timm model name: `tf_efficientnetv2_s.in21k_ft_in1k`. Adds yet more classifier diversity.

**Expected benefit:** +0.01-0.03 classification mAP
**Risk:** Low
**Estimated time:** 2-3 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="effnetv2-s-384px-classifier" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--epochs,50,--batch-size,48,--lr,2e-4,--model-name,tf_efficientnetv2_s.in21k_ft_in1k,--input-size,384,--gcs-name,classifier_effnetv2_s_384.pt,--run-id,effnetv2-s-384"
```

#### Job 9: YOLOv8l-1600 (higher resolution)
**Rationale:** 1600px pushes the resolution boundary. Shelf images are high-res, and 1600px means each product crop has ~25% more pixels than at 1280px. However, training is much slower and batch size must drop to 2.

**Expected benefit:** +0.01-0.02 detection mAP
**Risk:** High -- very slow training, may not converge in 150 epochs overnight; batch=2 can be unstable
**Estimated time:** 8-12 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="yolov8l-1600-highres" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--data,training/data.yaml,--model,yolov8l.pt,--imgsz,1600,--batch,2,--epochs,100,--run-id,yolov8l-1600-highres"
```

#### Job 10: ConvNeXt-Base classifier at 384px (larger model)
**Rationale:** If ConvNeXt-Small works well, the Base variant (~89M params) has more capacity for 356 fine-grained classes. State dict is ~340MB, which is too large for submission directly, but could be distilled or quantized.

**Expected benefit:** +0.02-0.04 classification mAP over Small
**Risk:** High -- 340MB model won't fit in 420MB budget alongside YOLO models unless quantized to int8 (~85MB). Quantization is untested.
**Estimated time:** 4-5 hours on A100

```bash
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name="convnext-base-384px-classifier" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
  --args="--epochs,50,--batch-size,32,--lr,1e-4,--model-name,convnext_base.fb_in22k_ft_in1k,--input-size,384,--gcs-name,classifier_convnext_base_384.pt,--run-id,convnext-base-384"
```

---

## Required Code Changes Before Launching

### Change 1: Add model-name, input-size, loss, and gcs-name args to train_classifier.py

The classifier script currently hardcodes `efficientnet_b3`, `INPUT_SIZE=300`, `CrossEntropyLoss`, and `GCS_WEIGHTS_NAME`. We need CLI args for all of these to run diverse classifier jobs without rebuilding the container each time.

**Args to add:**
- `--model-name` (default: `efficientnet_b3`) -- timm model name
- `--input-size` (default: 300) -- input resolution
- `--use-focal-loss` (flag) -- use focal loss instead of CrossEntropy
- `--gcs-name` (default: `classifier_efficientnet_b3.pt`) -- GCS upload filename

**Focal loss implementation** (add to train_classifier.py):
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Change 2: Add copy-paste and augmentation args to train.py

Add `--copy-paste` (float, default 0.0) arg and pass it to `model.train()`:
```python
parser.add_argument("--copy-paste", type=float, default=0.0)
# In model.train():
copy_paste=args.copy_paste,
```

### Change 3: Rebuild and push the container

After making code changes:
```bash
cd /workspaces/norgesgruppen_ai_vision
gcloud builds submit --config=cloudbuild.yaml --project=ai-nm26osl-1792
```

This must complete BEFORE launching any Tier 1-3 classifier jobs or copy-paste YOLO jobs. The Tier 1 YOLO retraining jobs (Jobs 1, 2) can launch immediately on the EXISTING container since they don't need new args.

---

## Launch Sequence

### Phase 1: Immediate (no code changes needed)

Launch on the EXISTING container image:

1. **Job 1: YOLOv8l-1280 corrected labels** -- launch NOW
2. **Job 2: YOLOv8x-1280 corrected labels** -- launch NOW
3. **Job 6: YOLOv8m-1280 corrected labels** -- launch NOW

These 3 YOLO jobs use the existing train.py with no modifications.

### Phase 2: After code changes + container rebuild (~30 min)

1. Modify `train_classifier.py` (add --model-name, --input-size, --use-focal-loss, --gcs-name)
2. Modify `train.py` (add --copy-paste)
3. Rebuild container: `gcloud builds submit --config=cloudbuild.yaml --project=ai-nm26osl-1792`
4. Launch classifier jobs:
   - **Job 3: ConvNeXt-Small 384px** -- highest priority classifier
   - **Job 4: EfficientNet-B3 384px focal** -- second classifier
   - **Job 5: Swin-Tiny 384px** -- third classifier
5. Launch augmentation YOLO job:
   - **Job 7: YOLOv8l-1280 copy-paste**

### Phase 3: If GPU slots available

6. **Job 8: EfficientNet-V2-Small 384px**
7. **Job 9: YOLOv8l-1600**
8. **Job 10: ConvNeXt-Base 384px** (only if we plan to quantize)

---

## Weight Budget Planning

Current submission: 3 YOLO models = 221MB, budget = 420MB, remaining = 199MB

**Best-case overnight ensemble (fits in 420MB):**

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv8l-1280 (corrected) | 85MB | Primary detector |
| YOLOv8x-1280 (corrected) | 132MB | Ensemble diversity |
| ConvNeXt-Small classifier | ~83MB | Two-stage classification |
| **Total** | **~300MB** | **Fits in 420MB** |

This drops to 2 YOLO models but adds the stronger x-1280. The classifier handles classification, making the third YOLO model less important.

**Alternative (3 YOLO + classifier):**

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv8l-1280 (corrected) | 85MB | Primary detector |
| YOLOv8l-640 (corrected) | 85MB | Multi-scale |
| YOLOv8m-1280 (corrected) | 51MB | Diversity |
| Classifier (EfficientNet-B3) | ~46MB | Classification |
| **Total** | **~267MB** | **Fits in 420MB** |

Note: The 3-weight-file limit in `validate_submission.sh` may need to be updated to allow classifier.pt as a 4th file, OR we store classifier.pt outside `weights/` dir.

---

## What NOT to Do

1. **Knowledge distillation** -- Too complex to implement, test, and debug in 2 days. Skip.
2. **CLIP/reference image similarity** -- Great idea but requires significant inference code changes and CLIP model (~350MB) doesn't fit in weight budget. The pre-computed embedding approach needs too much new code.
3. **YOLO11 or newer architectures** -- The container has ultralytics 8.1.0 which doesn't support YOLO11. Rebuilding with a newer ultralytics version risks breaking everything.
4. **Multi-GPU training** -- Adds complexity for marginal speedup on our small dataset.
5. **Training from scratch** -- Always fine-tune from COCO/ImageNet pretrained weights.

---

## Morning Checklist (when you return)

1. Check all job statuses:
   ```bash
   gcloud ai custom-jobs list --region=europe-west4 --project=ai-nm26osl-1792 --sort-by=~createTime --limit=15
   ```

2. Download all completed weights:
   ```bash
   gsutil ls gs://ai-nm26osl-1792-nmiai/weights/
   gsutil -m cp gs://ai-nm26osl-1792-nmiai/weights/*.pt weights/
   ```

3. Compare retrained YOLO models (corrected labels) vs originals:
   ```bash
   python scripts/eval_offline.py --weights weights/yolov8l-1280-corrected.pt --imgsz 1280
   python scripts/eval_offline.py --weights weights/yolov8x-1280-corrected.pt --imgsz 1280
   ```

4. Test classifier accuracy:
   - Download best classifier weights
   - Update `src/constants.py` with new model name and input size
   - Run full pipeline end-to-end
   - Compare with and without classifier

5. Build best submission config:
   - Pick top 2-3 YOLO models (stay within weight budget)
   - Pick best classifier
   - Update `src/constants.py`
   - Run `bash scripts/validate_submission.sh`
   - Submit

---

## Expected Overnight Outcome

**Conservative estimate:** Corrected-label YOLO models + best classifier deployed
- Detection: 0.82-0.85 mAP (up from ~0.80)
- Classification: 0.55-0.65 mAP (up from ~0.50)
- **Score: 0.74-0.79** (up from 0.7084)

**Optimistic estimate:** Best YOLO ensemble + ConvNeXt classifier at 384px
- Detection: 0.85-0.88 mAP
- Classification: 0.65-0.75 mAP
- **Score: 0.79-0.84**

**Stretch goal:** All optimizations land, good threshold tuning
- Detection: 0.88-0.90 mAP
- Classification: 0.75-0.85 mAP
- **Score: 0.84-0.89**

The gap to the top (0.92) likely requires techniques we don't have time for (better data, pseudo-labeling, test-time tricks), but reaching 0.80-0.85 would be a significant jump from rank 95 to potentially top 30-40.
