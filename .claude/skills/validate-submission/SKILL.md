# Validate Submission Skill

Use this skill before creating the final competition submission zip. Runs a complete pre-flight checklist to catch disqualifying issues before they reach the sandbox.

## Trigger

Use when:
- About to create a submission zip
- After any change to `run.py` or `src/`
- Before the competition deadline

## Checklist

### 1. Forbidden Import Scan (CRITICAL — disqualifies submission)

```bash
# Must return no matches
grep -rPn "^import os$|^import os |from os import|from os\.|^import subprocess|from subprocess|^import socket|from socket" src/ run.py
```

Expected: no output. If any match → STOP, fix before proceeding.

### 2. Entry Point Structure

```bash
# run.py must exist at repo root
test -f run.py && echo "OK: run.py exists at root" || echo "FAIL: run.py missing"

# Must accept --input and --output
python run.py --help | grep -E "\-\-input|\-\-output"
```

### 3. Weight File Size

```bash
# Total weights must be ≤ 420 MB
du -sh weights/ 2>/dev/null || echo "weights/ directory empty or missing"
find weights/ -name "*.pt" -o -name "*.onnx" -o -name "*.engine" 2>/dev/null | \
  xargs -I{} wc -c {} | awk '{total += $1} END {printf "Total weight size: %.1f MB\n", total/1024/1024}'
```

### 4. Output JSON Schema Validation

```bash
# Run on test fixtures and validate output
python run.py --input tests/fixtures/images/ --output /tmp/submission_test.json
python -c "
import json
from pathlib import Path
data = json.loads(Path('/tmp/submission_test.json').read_text())
assert isinstance(data, list), 'Output must be a JSON array'
for i, item in enumerate(data[:5]):
    assert 'image_id' in item, f'Missing image_id at index {i}'
    assert 'category_id' in item, f'Missing category_id at index {i}'
    assert 'bbox' in item, f'Missing bbox at index {i}'
    assert 'score' in item, f'Missing score at index {i}'
    assert isinstance(item['image_id'], int), f'image_id must be int at {i}'
    assert isinstance(item['category_id'], int), f'category_id must be int at {i}'
    assert isinstance(item['bbox'], list) and len(item['bbox']) == 4, f'bbox must be [x,y,w,h] at {i}'
    assert isinstance(item['score'], float), f'score must be float at {i}'
    assert 0 <= item['category_id'] <= 355, f'category_id out of range at {i}'
print(f'Schema OK: {len(data)} detections validated')
"
```

### 5. Inference Timing (extrapolation)

```bash
# Time inference on a small sample, extrapolate to full test set
# Assumes ~500 test images (adjust TEST_SET_SIZE if known)
TEST_SET_SIZE=500
SAMPLE_SIZE=10
python -c "
import time
from pathlib import Path
import torch
from ultralytics import YOLO
from src.constants import CONFIDENCE_THRESHOLD, MODEL_PATH

model = YOLO(MODEL_PATH)
model.to('cuda')
images = sorted(Path('tests/fixtures/images/').glob('*.jpg'))[:$SAMPLE_SIZE]
if not images:
    print('No fixture images found — skipping timing test')
else:
    t0 = time.perf_counter()
    with torch.no_grad():
        for p in images:
            model.predict(str(p), verbose=False, conf=CONFIDENCE_THRESHOLD)
    elapsed = time.perf_counter() - t0
    per_image = elapsed / len(images)
    projected = per_image * $TEST_SET_SIZE
    budget = 250
    status = 'OK' if projected < budget else 'WARNING: over budget'
    print(f'Per-image: {per_image*1000:.0f}ms | Projected total: {projected:.0f}s / 300s [{status}]')
"
```

### 6. Zip Structure Check

```bash
# If a zip already exists, verify run.py is at root (not in a subfolder)
ls submission*.zip 2>/dev/null | head -1 | xargs -I{} unzip -l {} | grep "run.py" | head -3
```

## Pass Criteria

All checks must pass:
- [ ] Zero forbidden imports
- [ ] `run.py` exists at root with `--input`/`--output` args
- [ ] Weight files ≤ 420 MB total
- [ ] Output JSON schema valid for all fields
- [ ] Projected inference time < 250s

## If a Check Fails

1. **Forbidden imports**: Replace `os.path.*` with `pathlib.Path.*`, remove subprocess/socket calls entirely
2. **Missing weights**: Pull from GCS — `gcloud storage cp gs://ai-nm26osl-1792-nmiai/weights/model.pt weights/`
3. **Weight size over limit**: Use `model.export(format='onnx', half=True)` to compress, or use YOLOv8s instead of YOLOv8m
4. **Schema errors**: Check that `int()` and `float()` casts are applied to all output fields
5. **Timing over budget**: Switch to TensorRT engine export or reduce `IMAGE_SIZE` in constants

## Gotchas

- `bbox` must be `[x, y, width, height]` NOT `[x1, y1, x2, y2]` — xyxy vs xywh conversion is a common mistake
- `score` must be Python `float`, not `numpy.float32` — use `float(box.conf[0].item())`
- `category_id` must be Python `int`, not `torch.Tensor` — use `int(box.cls[0].item())`
- Empty predictions are valid (no detections for an image simply produces no entries for that image_id)
