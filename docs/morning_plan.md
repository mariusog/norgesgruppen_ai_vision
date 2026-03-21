# Morning Action Plan -- Saturday March 21, 2026

**Written:** midnight Saturday by professor/strategist agent
**For:** lead-agent to execute step-by-step when user wakes up
**Current score:** 0.9095 (rank #23/238). Leader: 0.9221. Gap: 0.0126.
**Submissions available today:** 6 (Saturday). 3 more Sunday. Total: 9 remaining.
**Deadline:** Sunday evening, March 22.

---

## CRITICAL BUG -- FIX BEFORE ANY SUBMISSION

`scripts/create_submission.sh` line 43-44 only includes `src/__init__.py` and `src/constants.py` in the zip, but `run.py` line 59 imports `src/prototype_matcher.py`. If prototype matching is enabled (currently `USE_PROTOTYPE_MATCHING = True` in constants.py), the submission will crash on import.

**Fix options (pick one):**
1. Add `src/prototype_matcher.py` to the zip in `scripts/create_submission.sh` (change line 43 to `src/*.py`)
2. Guard the import in `run.py` so it only imports when the file exists and is needed

**Recommended:** Option 1 -- change `create_submission.sh` line 43-44 from:
```
  src/__init__.py \
  src/constants.py \
```
to:
```
  src/__init__.py \
  src/constants.py \
  src/prototype_matcher.py \
```

Also verify the py file count stays under 10 (currently 3 files: run.py, src/__init__.py, src/constants.py, src/prototype_matcher.py = 4 files, well within the 10-file limit).

---

## STEP 0: Check Overnight Training Results (15 minutes)

Run immediately upon waking:

```bash
# Check which training jobs completed
bash /workspaces/norgesgruppen_ai_vision/scripts/check_jobs.sh

# List all available classifier weights on GCS
gcloud storage ls gs://ai-nm26osl-1792-nmiai/weights/ | grep classifier
```

**Expected overnight results (assumptions for this plan):**
- Classifier v1 (EffNet-B3 224px): ~90% val accuracy -- DONE
- ConvNeXt-Small 384px: ~91-92% val accuracy -- DONE (this is our best)
- Classifier v2 (EffNet-B3 300px): ~89-90% -- DONE
- EffNet-B3 focal 384px: ~87-88% -- possibly done
- Swin-Tiny 224px: ~89% -- possibly still running
- EfficientNetV2-S 384px: ~89-90% -- possibly still running

**Decision tree:**
- If ConvNeXt >= 90%: deploy ConvNeXt as primary classifier (best single model)
- If ConvNeXt < 90%: deploy EffNet-B3 v1 as primary (safe choice at ~90%)
- For ensemble: pair ConvNeXt + EffNet-B3 (architectural diversity)

---

## STEP 1: Download Best Classifier Weights (10 minutes)

```bash
cd /workspaces/norgesgruppen_ai_vision

# Download ConvNeXt-Small (expected best, ~83MB)
gsutil cp gs://ai-nm26osl-1792-nmiai/weights/classifier_convnext_small.pt weights/classifier.pt

# Download EfficientNet-B3 v1 for ensemble (~46MB)
gsutil cp gs://ai-nm26osl-1792-nmiai/weights/classifier_efficientnet_b3.pt weights/classifier2.pt
```

**Note:** The filenames on GCS may differ. Adjust based on what `gcloud storage ls` shows. The key is:
- `weights/classifier.pt` = the single best classifier (ConvNeXt-Small if >= 90%)
- `weights/classifier2.pt` = second-best different-architecture classifier for ensemble

---

## STEP 2: Weight Budget Analysis (5 minutes)

**Hard constraint: max 3 weight files, max 420 MB total.**

This is the single most important constraint. We cannot ship 3 YOLO models + 2 classifiers + prototypes. We must choose.

### Configuration A: Best detection + single classifier (RECOMMENDED FIRST SUBMISSION)
| File | Size | Purpose |
|------|------|---------|
| weights/yolov8l-1280-corrected.pt | 85 MB | Primary detector |
| weights/yolov8x-1280-corrected.pt | 132 MB | Strong ensemble partner |
| weights/classifier.pt (ConvNeXt-Small) | ~83 MB | Best classifier |
| **Total** | **~300 MB** | 2 YOLO + 1 classifier |

Detection: 2-model ensemble (dual 1280px, corrected labels)
Classification: single ConvNeXt-Small
**This drops yolov8l-640-aug.pt from the current ensemble** to make room for the classifier.

### Configuration B: 2 YOLO + 2 classifiers (SECOND SUBMISSION)
| File | Size | Purpose |
|------|------|---------|
| weights/yolov8l-1280-corrected.pt | 85 MB | Primary detector |
| weights/yolov8x-1280-corrected.pt | 132 MB | Strong ensemble partner |
| weights/classifier.pt (ConvNeXt+EffNet merged) | ~129 MB | Both classifiers in one file |
| **Total** | **~346 MB** | Needs classifiers merged into single .pt |

**Problem:** 3-file limit. To use 2 classifiers with 2 YOLO models, we need to either:
- Merge both classifier state_dicts into a single .pt file (save as dict with two keys)
- OR drop to 1 YOLO model + 2 classifiers

**Merge approach (do this if ensemble shows gains on eval):**
```python
import torch
merged = {
    "classifier_0": {"model_name": "convnext_small.fb_in22k_ft_in1k", "state_dict": torch.load("weights/classifier.pt")},
    "classifier_1": {"model_name": "efficientnet_b3", "state_dict": torch.load("weights/classifier2.pt")},
}
torch.save(merged, "weights/classifiers_merged.pt")
```
Then update `load_classifier()` in `run.py` to handle the merged format.

### Configuration C: Keep current 3 YOLO + single classifier
| File | Size | Purpose |
|------|------|---------|
| weights/yolov8l-1280-corrected.pt | 85 MB | |
| weights/yolov8x-1280-corrected.pt | 132 MB | |
| weights/yolov8l-640-aug.pt | 85 MB | |
| **Total** | **302 MB** | No room for classifier in 3-file limit |

**This won't work** -- we need the classifier, but we're already at 3 files for YOLO. We MUST drop a YOLO model to add the classifier.

### Prototype embeddings
The `prototypes.pt` file (~2MB) counts toward the 3-file limit. **It must be bundled inside one of the classifier weight files** or we skip prototype matching.

**Decision:** Bundle prototypes inside the classifier .pt file:
```python
# After precomputing prototypes:
classifier_data = torch.load("weights/classifier.pt")
prototypes = torch.load("weights/prototypes.pt")
combined = {"state_dict": classifier_data, "prototypes": prototypes}
torch.save(combined, "weights/classifier.pt")
```
Then update `load_classifier()` and `load_prototypes()` to extract from the combined file.

---

## STEP 3: Update constants.py for Config A (10 minutes)

**File:** `/workspaces/norgesgruppen_ai_vision/src/constants.py`

Changes needed:

```python
# Line 57-61: Change ensemble to 2 models (drop l-640-aug)
ENSEMBLE_WEIGHTS: list[str] = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
]

# Line 65: Match sizes
ENSEMBLE_IMAGE_SIZES: list[int] = [1280, 1280]

# Line 79-80: Update classifier to ConvNeXt-Small
CLASSIFIER_PATH = "weights/classifier.pt"
CLASSIFIER_MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"

# Line 90: Update input size for ConvNeXt (384px native)
CLASSIFIER_INPUT_SIZE = 384

# Line 95: Lower the gate -- ConvNeXt at 91%+ should override almost always
CLASSIFIER_CONFIDENCE_GATE = 0.10

# Line 111: Disable prototype matching initially (test it after first submission)
USE_PROTOTYPE_MATCHING = False
```

---

## STEP 4: Run Eval Before Submitting (30-45 minutes)

Launch eval jobs on Vertex AI to compare configs. Run these in parallel:

```bash
# Eval A: 2 YOLO corrected + ConvNeXt classifier (our target config)
bash /workspaces/norgesgruppen_ai_vision/scripts/launch_eval.sh \
    "eval-2yolo-convnext" \
    "--ensemble-weights weights/yolov8l-1280-corrected.pt,weights/yolov8x-1280-corrected.pt --ensemble-sizes 1280,1280 --classifier weights/classifier.pt --classifier-model convnext_small.fb_in22k_ft_in1k --classifier-input-size 384 --classifier-gate 0.10"

# Eval B: Same but without classifier (to measure classifier delta)
bash /workspaces/norgesgruppen_ai_vision/scripts/launch_eval.sh \
    "eval-2yolo-noclassifier" \
    "--ensemble-weights weights/yolov8l-1280-corrected.pt,weights/yolov8x-1280-corrected.pt --ensemble-sizes 1280,1280 --no-classifier"

# Eval C: Current config (3 YOLO, no classifier) as baseline
bash /workspaces/norgesgruppen_ai_vision/scripts/launch_eval.sh \
    "eval-3yolo-baseline" \
    "--ensemble-weights weights/yolov8l-1280-corrected.pt,weights/yolov8x-1280-corrected.pt,weights/yolov8l-640-aug.pt --ensemble-sizes 1280,1280,640 --no-classifier"
```

**While waiting for eval results, proceed to Step 5.**

---

## STEP 5: Precompute Prototypes (20 minutes, while eval runs)

Even if we don't use prototypes in submission 1, precompute them now so they're ready.

```bash
cd /workspaces/norgesgruppen_ai_vision

python scripts/precompute_prototypes.py \
    --classifier-path weights/classifier.pt \
    --model-name convnext_small.fb_in22k_ft_in1k \
    --output weights/prototypes.pt
```

Check the output size: `ls -lh weights/prototypes.pt` (should be ~2MB).

---

## STEP 6: Submission 1 -- Config A (when eval results confirm gains)

**Expected: 2 YOLO corrected (1280px) + ConvNeXt classifier + TTA + score fusion**

1. Verify constants.py matches Config A from Step 3
2. Ensure only the right weight files are in `weights/`:
   ```bash
   cd /workspaces/norgesgruppen_ai_vision/weights
   # Keep only: yolov8l-1280-corrected.pt, yolov8x-1280-corrected.pt, classifier.pt
   # Move others out temporarily:
   mkdir -p /tmp/weight_backup
   mv model.pt yolov8l-1280-aug.pt yolov8l-640-aug.pt yolov8m-* yolov8x-1280-aug.pt yolov8x-640-aug.pt /tmp/weight_backup/
   ls -lh /workspaces/norgesgruppen_ai_vision/weights/
   ```
3. Fix `create_submission.sh` (add `src/prototype_matcher.py` -- see CRITICAL BUG above)
4. Validate and create submission:
   ```bash
   bash /workspaces/norgesgruppen_ai_vision/scripts/validate_submission.sh
   bash /workspaces/norgesgruppen_ai_vision/scripts/create_submission.sh
   ```
5. Submit the zip

**Expected outcome:** +0.005 to +0.015 over 0.9095 from the classifier alone. Target: 0.915-0.925.

---

## STEP 7: Analyze Eval Results and Plan Submission 2 (15 minutes)

Check eval job results:
```bash
gcloud ai custom-jobs list --region=us-central1 --project=ai-nm26osl-1792 --limit=5 --format='table(displayName,state)'
```

Stream logs of completed jobs to read mAP scores.

**Key questions from eval:**
1. How much did dropping the 3rd YOLO model (l-640-aug) cost in detection mAP?
2. How much did the ConvNeXt classifier gain in classification mAP?
3. What is the net effect on combined score?
4. Is classifier TTA helping (compare TTA vs no-TTA)?
5. What is the optimal `CLASSIFIER_CONFIDENCE_GATE` (0.05 vs 0.10 vs 0.15)?

---

## STEP 8: Submission 2 -- Classifier Ensemble or Prototype Matching (afternoon)

Based on eval results, pick ONE of these configs:

### Option A: Classifier Ensemble (if single classifier showed big gains)

Merge two classifiers into one file to stay within 3-file limit:
```python
# Run in Python:
import torch
state_dict_1 = torch.load("weights/classifier.pt", map_location="cpu")  # ConvNeXt
state_dict_2 = torch.load("/tmp/weight_backup/classifier2.pt", map_location="cpu")  # EffNet-B3
merged = {
    "models": [
        {"model_name": "convnext_small.fb_in22k_ft_in1k", "input_size": 384, "state_dict": state_dict_1},
        {"model_name": "efficientnet_b3", "input_size": 300, "state_dict": state_dict_2},
    ]
}
torch.save(merged, "weights/classifier.pt")
```

Then update `load_classifier()` in `run.py` to detect and handle the merged format. Update `CLASSIFIER_ENSEMBLE` in constants.py.

**Weight budget:** 85 + 132 + (83+46) = 346 MB. Within 420 MB.

### Option B: Add Prototype Matching (if classifier confidence distribution shows many mid-confidence predictions)

Bundle prototypes into classifier file:
```python
import torch
classifier_sd = torch.load("weights/classifier.pt", map_location="cpu")
prototypes = torch.load("weights/prototypes.pt", map_location="cpu")
combined = {"state_dict": classifier_sd, "prototypes": prototypes}
torch.save(combined, "weights/classifier.pt")
```

Update `load_classifier()` and prototype loading to extract from combined file.
Set `USE_PROTOTYPE_MATCHING = True` in constants.py.

### Option C: Restore 3rd YOLO model (if eval shows dropping it hurt detection too much)

If the classifier gain doesn't compensate for losing the 3rd YOLO model:
- Go back to 3 YOLO models (current sub 4 config = 0.9095)
- Bundle the classifier INSIDE one of the YOLO weight files (hacky but works)
- Or accept 2 YOLO + classifier if the net is positive

---

## STEP 9: Submission 3 -- Tuned Thresholds (late afternoon)

After submissions 1-2 give us signal:

1. Sweep `CLASSIFIER_CONFIDENCE_GATE`: {0.05, 0.10, 0.15, 0.20}
2. Sweep `SCORE_FUSION_ALPHA`: {0.5, 0.6, 0.7, 0.8, 0.9}
3. Sweep `WBF_IOU_THRESHOLD`: {0.50, 0.55, 0.60}
4. Check if `USE_CLASSIFIER_TTA = False` is better (saves time, sometimes TTA hurts)

Use `scripts/eval_full_pipeline.py` locally or via Vertex AI for each config. Pick the best combo and submit.

---

## STEP 10: Submissions 4-6 (evening)

**Submission 4:** Best config from today + prototype matching enabled
**Submission 5:** Alternative ensemble config (e.g., swap in different YOLO pair, or try EffNet-B3 alone if ConvNeXt disappointed)
**Submission 6:** Safety -- resubmit the best-scoring config from today unchanged

---

## CONTINGENCY PLANS

### If the classifier does NOT help (score drops or stays flat):

This would mean YOLO's classification is already good for the test distribution. In this case:

1. **Revert to 3-YOLO ensemble** (our proven 0.9095 config)
2. Focus remaining submissions on:
   - **WBF/NMS threshold tuning** -- sweep all thresholds for the corrected-label models
   - **Confidence threshold tuning** -- try `CONFIDENCE_THRESHOLD = 0.005` for higher recall
   - **Try yolov8m-1280-corrected as 3rd model** instead of yolov8l-640-aug (same resolution = better WBF fusion)
3. Consider `USE_TTA = True` with all models at 1280px if timing allows

### If the classifier HURTS the score:

1. Check if `CLASSIFIER_CONFIDENCE_GATE` is too low -- raise to 0.30 or 0.50 so it only overrides when very confident
2. Check if `SCORE_FUSION_ALPHA` is wrong -- try 0.9 (mostly keep YOLO score)
3. Check if the classifier is trained on different label conventions than the test set
4. Submit with `USE_CLASSIFIER = False` as a safety baseline

### If we can't get above 0.9095:

1. The current config IS our best. Protect it.
2. Use remaining submissions for low-risk threshold sweeps only
3. On Sunday, resubmit the best config as the final safety submission

### If timing budget is tight on the competition server:

1. Disable classifier TTA first (`USE_CLASSIFIER_TTA = False`)
2. Reduce `CLASSIFIER_INPUT_SIZE` to 224 (faster inference)
3. Drop to 1 YOLO model if desperate (yolov8x-1280-corrected is the strongest single model)

---

## PRIORITY SUMMARY

| Priority | Action | Expected Gain | Time |
|----------|--------|--------------|------|
| P0 | Fix create_submission.sh (prototype_matcher.py) | Prevents crash | 2 min |
| P0 | Download classifiers from GCS | Prerequisite | 10 min |
| P1 | Deploy ConvNeXt classifier (Config A) | +0.005-0.015 | 15 min |
| P1 | Run eval to validate before submitting | De-risks | 30-45 min |
| P1 | Submit 1: 2 YOLO + ConvNeXt | Target: 0.915+ | 10 min |
| P2 | Precompute prototypes | Ready for later | 20 min |
| P2 | Submit 2: classifier ensemble or prototypes | +0.003-0.010 | 1 hr |
| P3 | Threshold sweep | +0.002-0.005 | 1 hr |
| P3 | Submit 3: tuned thresholds | +0.002-0.005 | 10 min |
| P4 | Evening submissions 4-6 | Polish / safety | 2 hr |

---

## EXACT CONSTANTS.PY FOR FIRST SUBMISSION

Reference: `/workspaces/norgesgruppen_ai_vision/src/constants.py`

```python
# Line 57-61 -- 2-model ensemble
ENSEMBLE_WEIGHTS: list[str] = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8x-1280-corrected.pt",
]
ENSEMBLE_IMAGE_SIZES: list[int] = [1280, 1280]

# Line 79-80 -- ConvNeXt-Small classifier
CLASSIFIER_PATH = "weights/classifier.pt"
CLASSIFIER_MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"

# Line 90 -- 384px input for ConvNeXt
CLASSIFIER_INPUT_SIZE = 384

# Line 91 -- enabled
USE_CLASSIFIER = True

# Line 95 -- aggressive gate (classifier is much better than YOLO at classification)
CLASSIFIER_CONFIDENCE_GATE = 0.10

# Line 98 -- TTA enabled (flip augmentation is cheap)
USE_CLASSIFIER_TTA = True

# Line 103 -- score fusion
SCORE_FUSION_ALPHA = 0.7

# Line 111 -- prototypes OFF for first submission
USE_PROTOTYPE_MATCHING = False
```

All other constants remain unchanged from current values.

---

## KEY FILES

- `/workspaces/norgesgruppen_ai_vision/src/constants.py` -- all tuning parameters
- `/workspaces/norgesgruppen_ai_vision/run.py` -- inference pipeline
- `/workspaces/norgesgruppen_ai_vision/src/prototype_matcher.py` -- prototype matching module
- `/workspaces/norgesgruppen_ai_vision/scripts/create_submission.sh` -- zip builder (NEEDS FIX)
- `/workspaces/norgesgruppen_ai_vision/scripts/validate_submission.sh` -- pre-flight checks
- `/workspaces/norgesgruppen_ai_vision/scripts/launch_eval.sh` -- Vertex AI eval launcher
- `/workspaces/norgesgruppen_ai_vision/scripts/eval_full_pipeline.py` -- offline eval script
- `/workspaces/norgesgruppen_ai_vision/scripts/precompute_prototypes.py` -- prototype builder
- `/workspaces/norgesgruppen_ai_vision/scripts/check_jobs.sh` -- training job status
