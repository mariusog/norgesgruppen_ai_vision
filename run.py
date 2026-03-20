"""
NM i AI 2026 - NorgesGruppen Object Detection
Competition entry point.

Usage:
    python run.py --input /path/to/images/ --output /path/to/predictions.json

Output: JSON array of {image_id, category_id, bbox, score} dicts.
        bbox format: [x, y, width, height] in pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.constants import (
    CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    INFERENCE_BATCH_SIZE,
    IOU_THRESHOLD,
    MODEL_PATH,
)


def load_model() -> YOLO:
    """Load YOLOv8 model onto CUDA device."""
    weights = Path(MODEL_PATH)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    model = YOLO(str(weights))
    model.to("cuda")
    return model


def collect_images(input_dir: Path) -> list[Path]:
    """Return sorted list of image paths from the input directory."""
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_inference(model: YOLO, image_paths: list[Path]) -> list[dict]:
    """Run detection on all images in batches, returning competition-format predictions."""
    predictions: list[dict] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), INFERENCE_BATCH_SIZE):
            batch_paths = image_paths[i : i + INFERENCE_BATCH_SIZE]
            batch_strs = [str(p) for p in batch_paths]
            batch_ids = [int(p.stem.split("_")[-1]) for p in batch_paths]

            results = model.predict(
                batch_strs,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
            )

            for image_id, result in zip(batch_ids, results, strict=True):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(box.cls[0].item()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(box.conf[0].item()),
                        }
                    )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="NorgesGruppen grocery detector")
    parser.add_argument("--input", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    t_start = time.perf_counter()

    model = load_model()
    image_paths = collect_images(input_dir)

    predictions = run_inference(model, image_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions))

    elapsed = time.perf_counter() - t_start
    print(f"Processed {len(image_paths)} images in {elapsed:.1f}s → {len(predictions)} detections")


if __name__ == "__main__":
    main()
