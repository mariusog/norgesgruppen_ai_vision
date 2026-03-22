# Professor Review -- Saturday March 21, 2026

**Current score:** 0.9095 (rank #23/238). Leader: 0.9255. Gap: 0.016.
**Submissions remaining:** ~6 today + 3 Sunday = ~9 total.
**Deadline:** Sunday evening.

---

## 1. Bundle Crash Analysis

### What the bundle approach does (run.py lines 79-107)

The bundle file (`yolov8l-640-bundle.pt`, 170MB) is a `torch.save`'d dict with keys:
- `yolo_bytes`: raw bytes of the YOLO .pt file (84MB)
- `classifier_state_dict`: dict with 574 keys (EfficientNet-B3)
- `classifier_model_name`: `"efficientnet_b3"`

At load time, `load_ensemble_models()` detects the bundle, writes `yolo_bytes` to `/tmp/_tmp_yolo.pt`, loads it as YOLO, then deletes the temp file. The classifier is extracted separately in `load_classifier()`.

### Why it crashed (exit code 1)

I tested the extraction locally and the mechanics are sound -- the YOLO bytes extract and the classifier state_dict loads correctly. The crash was almost certainly caused by one of these issues:

**Most likely cause: `src/prototype_matcher.py` missing from the submission zip.**

`run.py` line 59 unconditionally imports:
```python
from src.prototype_matcher import load_prototypes, match_prototypes
```

But `create_submission.sh` lines 43-44 only include `src/__init__.py` and `src/constants.py`. The `prototype_matcher.py` file is NOT included in the zip. This causes an `ImportError` at startup, which would manifest as exit code 1 before any model loading occurs.

**Confirmation:** The file `_tmp_yolo.pt` (85MB) still exists in `weights/`, which means it was created locally during testing but never cleaned up -- this is not the crash source but is a red herring residue file.

**Secondary possible cause: memory.** Loading the full bundle (170MB) into RAM, then extracting 84MB of bytes, then writing to disk, then loading YOLO from disk -- this temporarily requires ~340MB of RAM just for the bundle. On 8GB RAM this should be fine, but combined with 2 other YOLO models already in VRAM, it could be tight. However, the `map_location="cpu"` in the bundle load makes this unlikely.

**Fix:** The import crash is the most probable cause. Fix `create_submission.sh` to include `src/prototype_matcher.py`, OR make the import conditional:
```python
try:
    from src.prototype_matcher import load_prototypes, match_prototypes
except ImportError:
    load_prototypes = None
    match_prototypes = None
```

The `create_submission.py` script (line 39) already includes `prototype_matcher.py` in `REQUIRED_FILES`. So only the `.sh` version is broken. **Use `create_submission.py` instead of `.sh` going forward.**

### Is the bundle approach sound?

Yes, technically sound. But it adds fragility:
- Requires writing to `/tmp` at runtime (should be safe but adds a failure mode)
- Doubles memory usage temporarily during load
- Adds code complexity
- If the competition sandbox restricts `/tmp` writes, it fails silently

**Verdict:** The bundle approach works but is unnecessary if we use Swin-Tiny (28MB). See below.

---

## 2. Exact Weight Budgets for All Viable 3-File Configs

**Hard constraint: exactly 3 weight files, <= 420MB total.**

### Current best config (0.9095 -- no classifier)

| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| yolov8l-640-aug.pt | 85 MB |
| **Total** | **302 MB** |

### Config A: 3 YOLO + Swin-Tiny BUNDLED (THE WINNING MOVE)

| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + Swin-Tiny) | ~114 MB |
| **Total** | **~331 MB** |

**This keeps ALL 3 YOLO models AND adds the best small classifier.** 89MB of headroom remaining. This is the configuration I recommend.

### Config B: 3 YOLO + EffNet-B3 BUNDLED

| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + EffNet-B3) | ~130 MB |
| **Total** | **~347 MB** |

Also fits. 73MB headroom.

### Config C: 3 YOLO + EfficientNetV2-S BUNDLED

| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| BUNDLE(yolov8l-640-aug + EffNetV2-S) | ~132 MB |
| **Total** | **~349 MB** |

Fits. 71MB headroom.

### Config D: 2 YOLO + Swin-Tiny (no bundle, simpler)

| File | Size |
|------|------|
| yolov8l-1280-corrected.pt | 85 MB |
| yolov8x-1280-corrected.pt | 132 MB |
| Swin-Tiny classifier | 28 MB |
| **Total** | **245 MB** |

Fits easily but loses the 3rd YOLO model. Previous test showed 2 YOLO + classifier scored 0.8939 (much worse), so the 3rd YOLO model is clearly important.

### Configs that DON'T fit

- 3 YOLO + ConvNeXt-Small (190MB): 302 + 190 = 492MB (OVER, even bundled would be 85+132+276 = 493MB)
- Any config with ConvNeXt-Small requires dropping to 2 YOLO models

---

## 3. Can We Fit 3 YOLO + Classifier in 3 Files?

**YES -- via bundling, and it is straightforward with Swin-Tiny or EffNet-B3.**

The bundle approach stores the classifier `state_dict` alongside the YOLO bytes in a single file. The key insight: Swin-Tiny at 28MB adds minimal overhead. Bundle it into the smallest YOLO file (yolov8l-640-aug at 85MB) to create a ~114MB combined file. Total: 331MB, well within 420MB.

**The critical fix needed:** The unconditional import of `prototype_matcher` must be resolved before any submission. Either:
1. Include `src/prototype_matcher.py` in the zip (use `create_submission.py` instead of `.sh`)
2. Make the import conditional

**Recommended action:** Create the bundle with Swin-Tiny and test it:

```python
import torch
from pathlib import Path

yolo_bytes = Path("weights/yolov8l-640-aug.pt").read_bytes()
classifier_sd = torch.load("weights/swin_tiny_classifier.pt", map_location="cpu")

bundle = {
    "yolo_bytes": yolo_bytes,
    "classifier_state_dict": classifier_sd,
    "classifier_model_name": "swin_tiny_patch4_window7_224",  # verify exact timm name
}
torch.save(bundle, "weights/yolov8l-640-bundle.pt")
```

---

## 4. Classifier Compression / Quantization

### Options analyzed

| Method | Size reduction | Accuracy impact | Feasibility |
|--------|---------------|-----------------|-------------|
| FP16 state_dict | ~50% (halves float size) | Negligible | Easy, 1 line |
| INT8 quantization | ~75% | 1-3% accuracy drop | Medium, needs calibration |
| Pruning | 20-50% | 1-5% drop | Hard, needs retraining |
| Knowledge distillation to smaller model | Variable | Variable | Already done (Swin-Tiny IS small) |

### FP16 conversion (recommended)

```python
import torch
sd = torch.load("weights/classifier.pt", map_location="cpu")
sd_half = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
torch.save(sd_half, "weights/classifier_fp16.pt")
# EffNet-B3: 44MB -> ~22MB
# Swin-Tiny: 28MB -> ~14MB
```

At inference, cast back to FP32 or run in FP16 (which we already do for YOLO). Accuracy loss is negligible for inference.

**However:** With Swin-Tiny at only 28MB, compression is unnecessary. The bundle fits comfortably at 331MB total. FP16 conversion would save another 14MB, bringing it to 317MB -- nice but not needed.

---

## 5. Swin-Tiny Analysis

**Swin-Tiny at 28MB with 91.0% val accuracy is the clear winner for this situation.**

| Classifier | Val Accuracy | Size | Bundle total (3 YOLO) |
|-----------|-------------|------|----------------------|
| Swin-Tiny 224px | 91.0% | 28 MB | 331 MB |
| EffNet-B3 300px | 90.2% | 44 MB | 347 MB |
| EfficientNetV2-S 384px | 90.7% | 46 MB | 349 MB |
| ConvNeXt-Small 384px | 91.0% | 190 MB | DOES NOT FIT with 3 YOLO |

Swin-Tiny matches ConvNeXt-Small's 91.0% accuracy at 15% of the size. It is the obvious choice.

**One concern:** Swin-Tiny uses 224px input while ConvNeXt uses 384px. For fine-grained grocery products where label text matters, higher resolution helps. But the val accuracy numbers already account for this -- 91.0% at 224px is still excellent.

**Configuration constants for Swin-Tiny:**
```python
CLASSIFIER_MODEL_NAME = "swin_tiny_patch4_window7_224"  # verify exact timm name
CLASSIFIER_INPUT_SIZE = 224
```

---

## 6. Should We Try 2 YOLO + TTA + Swin Classifier?

**No. The data is clear: this is a losing trade.**

The 2 YOLO + classifier submission scored **0.8939**, which is 0.0156 WORSE than 3 YOLO without classifier (0.9095). This means dropping the 3rd YOLO model cost more in detection mAP than the classifier gained in classification mAP.

Score = 0.7 * det_mAP + 0.3 * cls_mAP. For the classifier to compensate for a detection loss:
- If dropping 3rd YOLO costs 0.02 det_mAP: needs +0.047 cls_mAP from classifier
- A 91% classifier vs YOLO's ~70% classification = +0.21 cls_mAP improvement
- 0.21 * 0.3 = +0.063 score from classification
- 0.02 * 0.7 = -0.014 score from detection loss
- Net: +0.049

**Wait -- the math says 2 YOLO + classifier SHOULD win.** The fact that it scored 0.8939 (worse) suggests either:
1. The classifier used in that submission was bad (maybe wrong model name, wrong input size, or wrong state_dict format)
2. The 2 YOLO models used were weaker (not the l-1280 + x-1280 pair)
3. The classifier was crashing silently, reverting to YOLO categories
4. Score fusion (alpha=0.7) was hurting the ranking

**This warrants investigation.** If the previous 2-YOLO+classifier test used a broken classifier, then 2 YOLO + working Swin-Tiny could actually beat 3 YOLO alone. But given we can now BUNDLE to get 3 YOLO + classifier, this question is moot -- go with the bundle.

---

## 7. Threshold Safety Review

### Current thresholds:
- `WBF_IOU_THRESHOLD = 0.50` (lowered from 0.55)
- `IOU_THRESHOLD = 0.50` (NMS, raised from 0.45)
- `CONFIDENCE_THRESHOLD = 0.01`
- `WBF_SKIP_BOX_THRESHOLD = 0.001`

### Assessment:

**WBF IoU 0.50:** More aggressive fusion. For dense shelves this is reasonable -- it merges boxes that overlap 50%+ into a single fused box. Risk: could merge distinct adjacent products of the same category. The 0.9095 score was achieved with this value, so it is empirically validated. **SAFE.**

**NMS IoU 0.50:** Raised from 0.45. This keeps more overlapping boxes (requires 50% IoU to suppress). Good for dense shelves. But note: YOLO NMS happens BEFORE WBF, so raising this gives WBF more raw detections to fuse. This is correct behavior. **SAFE.**

**Confidence 0.01:** Very low, maximizes recall. Correct for mAP evaluation which needs the full precision-recall curve. **SAFE.**

**WBF skip threshold 0.001:** Very permissive -- keeps almost all boxes for WBF. Correct approach. **SAFE.**

**Verdict:** These thresholds are all reasonable for dense shelf detection with WBF ensemble. No changes needed.

---

## 8. Code Bugs and Issues

### BUG 1 (CRITICAL): `prototype_matcher.py` not in submission zip

`create_submission.sh` lines 43-44 omit `src/prototype_matcher.py`, but `run.py` line 59 unconditionally imports it. Any submission built with the `.sh` script will crash immediately with `ImportError`.

**Fix:** Use `create_submission.py` (which already includes it) OR add `src/prototype_matcher.py` to the `.sh` script.

### BUG 2 (MEDIUM): Residual `_tmp_yolo.pt` in weights/

`weights/_tmp_yolo.pt` (85MB) is a leftover from bundle testing. It will be included in any submission zip built with the wildcard `weights/*.pt`. This:
- Wastes 85MB of the 420MB budget
- May confuse the model loading code
- Counts toward the 3-file weight limit

**Fix:** Delete `weights/_tmp_yolo.pt` immediately:
```bash
rm weights/_tmp_yolo.pt
```

### BUG 3 (MEDIUM): `model.pt` still in weights/

`weights/model.pt` (85MB) is listed as a duplicate. If included in the zip via wildcard, it wastes budget and may exceed the 3-file limit.

**Fix:** Either delete it or ensure the submission script excludes it. The `.py` script reads from `ENSEMBLE_WEIGHTS` so it only includes the right files. The `.sh` script includes ALL `*.pt` files -- dangerous.

### BUG 4 (LOW): `create_submission.sh` includes ALL .pt files

Line 45: `weights/*.pt` includes every .pt file in the directory. With the current weights/ directory containing 15 .pt files totaling 1.3GB, this creates a massively oversized invalid submission.

**Fix:** The `.sh` script is fundamentally broken for the current setup. Use `create_submission.py` exclusively, which reads `ENSEMBLE_WEIGHTS` from constants and only includes those specific files.

### BUG 5 (LOW): `weights_only=True` in prototype_matcher but `weights_only=False` patched globally

`run.py` line 24 patches `torch.load` to use `weights_only=False` globally. But `prototype_matcher.py` line 30 explicitly passes `weights_only=True`. The global patch may or may not override the explicit kwarg (depends on how `functools.partial` handles conflicting kwargs). In CPython, explicit kwargs override partial kwargs, so `weights_only=True` should win. But this is fragile.

**Risk:** Low, but worth being aware of. If prototypes fail to load, this could be why.

### BUG 6 (LOW): `BUNDLE_WEIGHT_PATH = ""` but bundle code still runs

In `load_classifier()` line 140: `bundle_path = Path(BUNDLE_WEIGHT_PATH) if BUNDLE_WEIGHT_PATH else None`. When `BUNDLE_WEIGHT_PATH = ""`, this correctly sets `bundle_path = None`. But if someone sets it to a non-existent path, it will try to load and fail. Not currently a bug but a latent issue.

### ISSUE 7 (STYLE): Unnecessary `_tmp_yolo.pt` hardcoded path

The bundle code writes to `/tmp/_tmp_yolo.pt` (line 97). If two inference processes run simultaneously (unlikely but possible), they would clobber each other. Use `tempfile.NamedTemporaryFile` instead. Note: `tempfile` is not in the blocked imports list.

---

## 9. Recommended Action Plan

### Priority 1: Create and test 3 YOLO + Swin-Tiny bundle (30 minutes)

This is the single highest-value action. It gives us everything:
- All 3 YOLO models (proven detection quality)
- TTA enabled (proven +0.01-0.03)
- Swin-Tiny classifier at 91% accuracy (classification boost)
- Total: ~331MB, well within 420MB

Steps:
1. Download Swin-Tiny weights from GCS (if not already local)
2. Create bundle: `yolov8l-640-aug.pt` + Swin-Tiny state_dict
3. Update `constants.py`:
   - `BUNDLE_WEIGHT_PATH = "weights/yolov8l-640-bundle.pt"`
   - `CLASSIFIER_MODEL_NAME = "swin_tiny_patch4_window7_224"`
   - `CLASSIFIER_INPUT_SIZE = 224`
   - `USE_CLASSIFIER = True`
4. Delete `_tmp_yolo.pt` and `model.pt` from weights/
5. Test locally (at least verify no crash)
6. Build submission with `create_submission.py` (NOT `.sh`)

### Priority 2: If Swin-Tiny unavailable, use EffNet-B3 bundle (already exists!)

The existing `yolov8l-640-bundle.pt` (170MB) already contains EffNet-B3. Just:
1. Set `BUNDLE_WEIGHT_PATH = "weights/yolov8l-640-bundle.pt"` in constants
2. Update `ENSEMBLE_WEIGHTS` to include the bundle path instead of `yolov8l-640-aug.pt`
3. Set `USE_CLASSIFIER = True`
4. Submit

### Priority 3: If bundle approach still crashes, fall back to 2 YOLO + classifier

Use the strongest 2-model pair (l-1280 + x-1280) plus the best available classifier. But first investigate WHY the previous 2-YOLO attempt scored so poorly -- see section 6 analysis.

### Priority 4: Threshold tuning (only after classifier is working)

Once we have a working 3-YOLO + classifier submission:
- Sweep `CLASSIFIER_CONFIDENCE_GATE`: {0.05, 0.10, 0.15, 0.20, 0.30}
- Sweep `SCORE_FUSION_ALPHA`: {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
- Consider `USE_CLASSIFIER_TTA = False` as a baseline (TTA may not help at 224px)

---

## 10. Expected Score Improvement

With 3 YOLO + Swin-Tiny (91% accuracy) + TTA:

- Detection: same as current (0.9095 baseline includes detection contribution)
- Classification: Swin-Tiny at 91% accuracy vs YOLO's estimated ~70% classification
- The 30% classification weight means: +0.21 cls_acc * 0.3 weight = +0.063 maximum theoretical gain
- Realistically, with confidence gating and imperfect crops: +0.01 to +0.03

**Conservative estimate:** 0.920-0.925
**Optimistic estimate:** 0.930+
**This would put us in the top 10.**

---

## Summary of Critical Actions

| # | Action | Impact | Time |
|---|--------|--------|------|
| 1 | Delete `_tmp_yolo.pt` and `model.pt` from weights/ | Prevents invalid submission | 1 min |
| 2 | Download Swin-Tiny classifier from GCS | Prerequisite | 5 min |
| 3 | Create Swin-Tiny bundle with yolov8l-640-aug | Enables 3 YOLO + classifier | 10 min |
| 4 | Update constants.py for bundle config | Configuration | 5 min |
| 5 | Test locally (no crash) | Validation | 10 min |
| 6 | Build zip with `create_submission.py` | Correct packaging | 5 min |
| 7 | Submit | Score improvement | 5 min |

**Total time to first submission: ~40 minutes.**
