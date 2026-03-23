# Strategy Plan -- 2026-03-20

## Situation Assessment

### What we have
- **5 weight files** on disk (434MB total), but `model.pt` = `yolov8l-640-aug.pt` (identical MD5)
- **4 unique models**: yolov8l-1280-aug (85MB), yolov8l-640-aug (85MB), yolov8m-640-aug (51MB), yolov8x-640-aug (132MB)
- **Current config**: 3-model WBF ensemble (l-1280, l-640, m-640) + TTA + FP16
- **EfficientNet-B3 classifier**: still training on A100, weights not available yet
- **No offline eval results**: `docs/eval_results.json` does not exist, no benchmark timing data
- **Deadline**: March 22, 2026 (2 days, 6 submissions remaining)
- **No competition scores yet** (no `memory/session_handoff.md`)

### Hard Constraints
| Constraint | Limit | Current |
|------------|-------|---------|
| Total weight size | 420 MB | Depends on config |
| Max weight files | **3** | 3 (current ensemble) |
| Inference timeout | 300s on L4 | **Unknown -- never benchmarked** |
| Submissions/day | 3 | 6 remaining (3 today + 3 tomorrow) |
| Pre-installed pkgs | ultralytics 8.1.0, torch 2.6.0, ensemble-boxes, timm | All used |

### Critical Risk: Timing is Unknown
We have **zero** benchmark data. The benchmark_results.md is empty. We are flying blind on the most important constraint. The current 3-model ensemble with TTA could easily exceed 300s.

**Estimated per-image timing on L4 (FP16):**

| Config | Per-image est. | 500-img est. | Fits 300s? |
|--------|---------------|-------------|------------|
| Single l-640, no TTA | ~25ms | ~12.5s | Yes |
| Single l-1280, no TTA | ~80ms | ~40s | Yes |
| Single x-640, no TTA | ~40ms | ~20s | Yes |
| 3-model ensemble (l-1280+l-640+m-640), no TTA | ~125ms | ~62.5s | Yes |
| 3-model ensemble + TTA (current) | ~350ms | ~175s | Probably yes |
| 3-model ensemble + TTA + classifier | ~400ms | ~200s | Risky |

These are rough estimates based on typical YOLO inference times. The actual numbers depend on image resolution, number of detections, and WBF overhead. **We must benchmark before submitting.**

Note: The ensemble processes images **one at a time** (no batching), which is suboptimal but avoids OOM risk on 8GB RAM.

---

## Decision Matrix

### Q1: What is the optimal model combination?

**Answer: 3-model ensemble (l-1280, l-640, m-640) is the best we can do.**

Rationale:
- Max 3 weight files enforced by `scripts/validate_submission.sh`
- l-1280 (85MB) + l-640 (85MB) + m-640 (51MB) = **221MB** -- well within 420MB
- Cannot add x-640 as a 4th file (would violate 3-file limit)
- Replacing m-640 (51MB) with x-640 (132MB) is an option: l-1280 + l-640 + x-640 = 302MB
- However, ensemble diversity matters more than individual model size. l + l + m gives architecture diversity (large vs medium backbone). l + l + x gives only scale diversity (same architecture family).
- **Recommendation**: Keep current l-1280 + l-640 + m-640 as the primary config. Test l-1280 + l-640 + x-640 as an alternative if we have a spare submission.

### Q2: Should we use TTA?

**Answer: Yes, but only if benchmarking confirms it fits in 300s.**

Rationale:
- TTA typically adds +1-3% mAP for ~2-3x time cost
- For a 3-model ensemble at 640/1280, estimated total ~175s with TTA -- likely fits
- But we MUST verify with actual benchmark before submitting
- **Fallback**: If TTA pushes us over 250s projected, disable it. The ensemble alone provides multi-scale views.

### Q3: Should we add yolov8x-640-aug.pt?

**Answer: Not as a 4th model (blocked). Only as a replacement for m-640.**

- 4 models exceeds the 3 weight file limit
- Could replace m-640: l-1280 + l-640 + x-640 = 302MB, 3 files
- x-640 is slower (~40ms vs ~20ms for m-640) which eats into TTA budget
- Worth testing as Alternative Submission B if we have submissions to spare

### Q4: Submission strategy for remaining 2 days?

**Answer: Conservative-first, then optimize.**

### Q5: Should we refactor ensemble to use batching?

**Answer: No. Too risky 2 days before deadline.**

- Current per-image WBF loop is correct and safe
- Batching would require rewriting the core inference loop
- Risk of OOM on 8GB RAM with batched multi-model inference
- Marginal time savings (~10-20%) not worth the regression risk
- The single-image loop is actually more memory-efficient for large images

---

## Improvement Plan

### Current: Unknown mAP | Target: Maximize | Gap: Unknown

### Phase 1: Critical -- Benchmark and Validate (Today, < 2 hours)

**Priority: HIGHEST. Without this, every submission is a gamble.**

#### 1.1 Run offline evaluation on val set
- **File**: `scripts/eval_offline.py`
- **Commands** (run sequentially on any available GPU):
  ```
  python scripts/eval_offline.py --weights weights/yolov8l-640-aug.pt --imgsz 640 --conf 0.01 --iou 0.45
  python scripts/eval_offline.py --weights weights/yolov8l-1280-aug.pt --imgsz 1280 --conf 0.01 --iou 0.45
  python scripts/eval_offline.py --weights weights/yolov8m-640-aug.pt --imgsz 640 --conf 0.01 --iou 0.45
  python scripts/eval_offline.py --weights weights/yolov8x-640-aug.pt --imgsz 640 --conf 0.01 --iou 0.45
  ```
- **Goal**: Get baseline mAP for each model. Know which model is strongest alone.
- **Expected**: l-1280 > x-640 > l-640 > m-640

#### 1.2 Time the ensemble pipeline
- **File**: No file changes needed, just run inference on sample images
- **Command**:
  ```
  time python run.py --input /path/to/val/images/ --output /tmp/test_pred.json
  ```
- **Goal**: Get actual per-image timing for current config (3-model + TTA)
- **Action**: If projected > 250s for 500 images, disable TTA in `src/constants.py` (`USE_TTA = False`)

#### 1.3 Validate submission structure
- **Command**: `bash scripts/validate_submission.sh`
- **Goal**: Ensure we pass all pre-flight checks before wasting a submission

### Phase 2: Optimize Parameters (Today, ~1 hour)

#### 2.1 Tune WBF thresholds
- **File**: `src/constants.py`
- **Current**: `WBF_IOU_THRESHOLD = 0.55`, `WBF_SKIP_BOX_THRESHOLD = 0.001`
- **Action**: Test WBF_IOU_THRESHOLD in {0.45, 0.50, 0.55, 0.60} on val set
- **Expected gain**: +0.5-1.5% mAP from better fusion threshold
- **How**: Use `scripts/sweep_thresholds.py` if it supports WBF params, or modify and run manually

#### 2.2 Tune confidence threshold
- **File**: `src/constants.py`
- **Current**: `CONFIDENCE_THRESHOLD = 0.01`
- **Action**: This is already very low (good for mAP recall curve). Leave as-is unless eval shows issues.
- **Note**: 0.01 is correct for mAP evaluation -- lower thresholds let the P-R curve extend further.

#### 2.3 Tune NMS IoU threshold
- **File**: `src/constants.py`
- **Current**: `IOU_THRESHOLD = 0.45`
- **Action**: Shelf images have dense, overlapping products. Test 0.50 and 0.55 -- higher thresholds keep more overlapping boxes, which may be correct for shelves.
- **Expected gain**: +0.5-1.0% mAP

### Phase 3: Classifier Integration (Tomorrow, if weights arrive)

#### 3.1 Download classifier weights
- **File**: `weights/classifier.pt`
- **Action**: Check if EfficientNet-B3 training has completed on A100 in us-central1. Download if ready.
- **Command**: `gsutil cp gs://YOUR_GCS_BUCKET/weights/classifier*.pt weights/`
- **Note**: classifier.pt does NOT count toward the 3 weight file limit IF the validate script only counts `.pt` files in `weights/` that match YOLO patterns. Check this.

#### 3.2 Test classifier impact
- **File**: `src/constants.py`
- **Current**: `USE_CLASSIFIER = True`, `CLASSIFIER_CONFIDENCE_GATE = 0.5`
- **Action**: Run inference with and without classifier, compare classification mAP
- **Expected gain**: +2-5% on classification mAP component (30% of score)
- **Risk**: Classifier may hurt detection speed. Budget ~50ms per image for crop-and-classify.

#### 3.3 Tune classifier confidence gate
- **File**: `src/constants.py`
- **Action**: Test gate values {0.3, 0.5, 0.7, 0.9}. Higher gate = fewer overrides = safer but less impact.

### Phase 4: Alternative Configs (Tomorrow, use remaining submissions)

#### 4.1 Test x-640 replacement
- **File**: `src/constants.py`
- **Change**:
  ```python
  ENSEMBLE_WEIGHTS = [
      "weights/yolov8l-1280-aug.pt",
      "weights/yolov8l-640-aug.pt",
      "weights/yolov8x-640-aug.pt",
  ]
  ENSEMBLE_IMAGE_SIZES = [1280, 640, 640]
  ```
- **Weight total**: 85 + 85 + 132 = 302MB (fits 420MB, 3 files)
- **Expected**: x-640 is a stronger individual model than m-640, but less diverse
- **Submit only if**: Phase 1 eval shows x-640 >> m-640 on val set

---

## Submission Strategy

### Today (March 20) -- 3 submissions available

| # | Config | Purpose | Risk |
|---|--------|---------|------|
| 1 | **3-model ensemble (l-1280+l-640+m-640) + TTA** | Safe baseline, current config | Medium (untested timing) |
| 2 | **Same + tuned WBF/NMS thresholds** | Optimize after seeing Sub 1 score | Low |
| 3 | **Hold** or use for x-640 variant if Sub 1 times out | Insurance | -- |

**Decision tree for Submission 1:**
- If benchmark shows projected time < 250s for 500 images: Submit with TTA
- If projected time 250-280s: Submit without TTA (`USE_TTA = False`)
- If projected time > 280s: Drop to 2-model ensemble (l-1280 + l-640 only) or single l-1280

### Tomorrow (March 21, final day) -- 3 submissions available

| # | Config | Purpose | Risk |
|---|--------|---------|------|
| 4 | **Best config from today + classifier** (if weights ready) | Classification mAP boost | Medium |
| 5 | **Alternative ensemble** (l-1280+l-640+x-640) or single best model | Diversify | Medium |
| 6 | **Final best** -- whatever scored highest, re-submit with any last tuning | Lock in best score | Low |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ensemble + TTA exceeds 300s | Disqualification (timeout) | Benchmark first; have fallback config without TTA |
| ensemble-boxes not pre-installed in sandbox | Runtime crash | CLAUDE.md says pre-installed; but verify in competition docs |
| timm not pre-installed in sandbox | Classifier fails | Classifier has graceful fallback (returns None) |
| Classifier hurts more than helps | Lower classification mAP | Gate at high confidence (0.7+); test on val first |
| Val set mAP doesn't correlate with test set | Misleading optimization | Submit safe config first; don't over-tune to val |
| Weight file count check fails | Submission rejected | Only include 3 YOLO weights; classifier may count as 4th |

### Weight File Count -- Critical Check

The validate script counts ALL `.pt` files in `weights/`. If we include `classifier.pt`, that's **4 .pt files** which FAILS the 3-file limit. Options:
1. Store classifier outside `weights/` directory (e.g., root or `src/`)
2. Remove `model.pt` (it's a duplicate of `yolov8l-640-aug.pt`)
3. Check if competition actually enforces the 3-file limit or if it's our own guard

**Recommended**: Remove `model.pt` from submission zip (it's redundant). This frees a slot for `classifier.pt`.

---

## Concrete File Changes Needed

### Immediate (before first submission)

1. **`src/constants.py`** -- Only change if benchmark shows timing issues:
   - `USE_TTA = False` (if ensemble + TTA > 250s projected)
   - Keep all other values as-is for first submission

2. **`weights/`** -- For submission zip, include exactly:
   - `yolov8l-1280-aug.pt` (85MB)
   - `yolov8l-640-aug.pt` (85MB)
   - `yolov8m-640-aug.pt` (51MB)
   - Total: 221MB (well within 420MB)
   - Do NOT include `model.pt` (duplicate) or `yolov8x-640-aug.pt` (4th file)

3. **`src/constants.py`** -- Update MODEL_PATH fallback:
   ```python
   MODEL_PATH = "weights/yolov8l-640-aug.pt"  # was "weights/model.pt"
   ```
   This ensures single-model fallback uses the correct file if ensemble is disabled.

### After Phase 1 benchmarking

4. **`src/constants.py`** -- Update thresholds based on eval results:
   - `WBF_IOU_THRESHOLD` -- adjust if sweep shows better value
   - `IOU_THRESHOLD` -- adjust if sweep shows better value

### For classifier integration (Phase 3)

5. **`src/constants.py`** -- If classifier weights arrive:
   - `CLASSIFIER_PATH = "classifier.pt"` (move to root, not weights/)
   - `CLASSIFIER_CONFIDENCE_GATE` -- tune based on val results

6. **`scripts/create_submission.sh`** -- Ensure zip includes classifier.pt at root if used

---

## Summary: Priority-Ordered Action Items

1. **BENCHMARK NOW** -- Run eval_offline.py on all 4 models + time the ensemble pipeline
2. **FIX MODEL_PATH** -- Change to `yolov8l-640-aug.pt` (model.pt is redundant)
3. **SUBMIT #1** -- 3-model ensemble, TTA on/off based on benchmark, current thresholds
4. **TUNE** -- Sweep WBF/NMS thresholds on val set
5. **SUBMIT #2** -- Best thresholds from sweep
6. **CLASSIFIER** -- Check if training finished, integrate if ready
7. **SUBMIT #3-6** -- Remaining submissions for classifier config and alternatives
