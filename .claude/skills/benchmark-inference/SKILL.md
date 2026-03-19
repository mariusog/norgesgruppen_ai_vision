# Benchmark Inference Skill

Use this skill to measure inference performance and verify the submission stays within the 300-second budget on the competition's NVIDIA L4 GPU.

## Trigger

Use when:
- After any change to `run.py`, `src/inference.py`, or model export format
- After changing `IMAGE_SIZE`, `CONFIDENCE_THRESHOLD`, or batch settings in `src/constants.py`
- Before creating a submission zip
- model-agent delivers new weights

## Benchmark Procedure

### Step 1: Profile a sample (p50/p95/p99 per-image latency)

```python
# Save as scripts/benchmark.py and run:
# python scripts/benchmark.py --images tests/fixtures/images/ --n 20
import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.constants import CONFIDENCE_THRESHOLD, MODEL_PATH

WARMUP_RUNS = 3


def benchmark(image_dir: Path, n_images: int) -> None:
    model = YOLO(MODEL_PATH)
    model.to("cuda")

    images = sorted(image_dir.glob("*.jpg"))[:n_images]
    if not images:
        print("No images found in fixture directory")
        return

    # Warmup
    with torch.no_grad():
        for p in images[:WARMUP_RUNS]:
            model.predict(str(p), verbose=False, conf=CONFIDENCE_THRESHOLD)

    # Timed runs
    latencies: list[float] = []
    with torch.no_grad():
        for p in images:
            t0 = time.perf_counter()
            model.predict(str(p), verbose=False, conf=CONFIDENCE_THRESHOLD)
            latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    mean = sum(latencies) / n

    # Extrapolate to full test set
    TEST_SET_SIZES = [100, 300, 500, 1000]
    print(f"\n=== Inference Benchmark ===")
    print(f"Sample: {n} images | Mean: {mean*1000:.0f}ms | p50: {p50*1000:.0f}ms | p95: {p95*1000:.0f}ms | p99: {p99*1000:.0f}ms")
    print(f"\nProjected total time (300s budget):")
    for size in TEST_SET_SIZES:
        projected = mean * size
        status = "OK" if projected < 250 else "OVER BUDGET"
        print(f"  {size:>5} images: {projected:>6.0f}s [{status}]")
```

### Step 2: Log results to docs/benchmark_results.md

After each run, append a row:

```markdown
| Date | Model | IMAGE_SIZE | FP16 | Engine | Mean (ms) | p95 (ms) | 500-img est (s) | Status |
|------|-------|-----------|------|--------|-----------|----------|-----------------|--------|
| 2026-03-19 | yolov8m | 640 | No | PyTorch | 450 | 620 | 225 | OK |
```

### Step 3: GPU memory check

```bash
python -c "
import torch
torch.cuda.empty_cache()
# Run one inference to populate memory stats
from pathlib import Path
from ultralytics import YOLO
from src.constants import MODEL_PATH, CONFIDENCE_THRESHOLD
model = YOLO(MODEL_PATH)
model.to('cuda')
images = list(Path('tests/fixtures/images/').glob('*.jpg'))
if images:
    with torch.no_grad():
        model.predict(str(images[0]), verbose=False, conf=CONFIDENCE_THRESHOLD)
print(torch.cuda.memory_summary(abbreviated=True))
"
```

## Optimization Decision Tree

```
Is p50 latency > 1000ms?
├── YES: Try TensorRT export (biggest win on L4)
│   model.export(format='engine', half=True, device=0)
│   Then reload: model = YOLO('weights/model.engine')
└── NO: Is p95 > 1500ms?
    ├── YES: Try half-precision (FP16)
    │   model.predict(..., half=True)
    └── NO: Is projected 500-img time > 200s?
        ├── YES: Try batch inference (batch=4 if VRAM allows)
        └── NO: Budget OK — proceed to submission
```

## Pass Criteria

Log to `docs/benchmark_results.md`. Task complete when:
- Mean per-image latency ≤ 1000ms
- Projected 500-image time ≤ 250s (50s safety margin)
- GPU peak memory ≤ 20 GB

## Gotchas

- Always run 3+ warmup inferences before timing — CUDA JIT compilation skews first results
- TensorRT `.engine` files are GPU-architecture specific — export on the target L4 GPU, not locally
- `half=True` (FP16) can slightly reduce mAP — always re-validate mAP after enabling
- Benchmark on representative images (similar resolution to test set), not tiny thumbnails
