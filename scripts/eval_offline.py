"""Offline score estimation using validation set.

Simulates the competition scoring formula:
  Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Uses ultralytics built-in validation which computes both metrics.

Usage:
    python scripts/eval_offline.py --weights weights/model.pt
    python scripts/eval_offline.py --weights weights/model.pt --imgsz 1280
    python scripts/eval_offline.py --weights weights/model.pt --conf 0.15 --iou 0.5
"""
from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import torch

torch.load = functools.partial(torch.load, weights_only=False)

from ultralytics import YOLO  # noqa: E402


def evaluate(
    weights: str,
    data: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> None:
    """Run validation and compute estimated competition score."""
    model = YOLO(weights)

    print(f"Evaluating: {weights}")
    print(f"  imgsz={imgsz}, conf={conf}, iou={iou}")
    print()

    t_start = time.perf_counter()
    with torch.no_grad():
        metrics = model.val(
            data=data,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=True,
            plots=False,
        )
    elapsed = time.perf_counter() - t_start

    # Extract metrics
    # model.val() computes mAP across all classes with correct category matching
    # This IS classification mAP (IoU >= 0.5 AND correct category)
    cls_map50 = float(metrics.box.map50)
    cls_map5095 = float(metrics.box.map)

    # For detection mAP (category-agnostic), we'd need to re-run with single class
    # Approximation: detection mAP >= classification mAP (finding is easier than naming)
    # Better: run val with single_cls=True for detection-only mAP
    print("\n--- Running detection-only evaluation (single_cls) ---")
    with torch.no_grad():
        det_metrics = model.val(
            data=data,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            plots=False,
            single_cls=True,
        )

    det_map50 = float(det_metrics.box.map50)

    # Competition score formula
    score = 0.7 * det_map50 + 0.3 * cls_map50

    print()
    print("=" * 50)
    print(f"  Detection mAP@0.5:       {det_map50:.4f}")
    print(f"  Classification mAP@0.5:  {cls_map50:.4f}")
    print(f"  Classification mAP@0.5-0.95: {cls_map5095:.4f}")
    print()
    print(f"  Estimated competition score:")
    print(f"    0.7 × {det_map50:.4f} + 0.3 × {cls_map50:.4f} = {score:.4f}")
    print("=" * 50)
    print(f"  Evaluation time: {elapsed:.1f}s")
    print()
    print("NOTE: This is on the validation set (38 images).")
    print("Competition uses a different test set — actual score will vary.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline score estimation")
    parser.add_argument("--weights", type=str, default="weights/model.pt")
    parser.add_argument("--data", type=str, default="training/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()
