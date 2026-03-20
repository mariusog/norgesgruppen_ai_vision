"""Sweep confidence and IOU thresholds on validation set to find optimal values.

Usage:
    python scripts/sweep_thresholds.py --weights weights/model.pt

Requires validation images + labels in training/data/ (run download_dataset.sh first).
Outputs best threshold combination to stdout.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from ultralytics import YOLO


CONF_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
IOU_VALUES = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]


def sweep(weights_path: str, data_yaml: str, imgsz: int) -> None:
    """Run validation at each threshold combination and report best."""
    model = YOLO(weights_path)

    best_map50 = 0.0
    best_conf = 0.25
    best_iou = 0.45
    results_table: list[tuple[float, float, float, float]] = []

    total = len(CONF_VALUES) * len(IOU_VALUES)
    print(f"Sweeping {total} combinations...")
    print(f"{'conf':>6} {'iou':>6} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 34)

    for conf, iou in itertools.product(CONF_VALUES, IOU_VALUES):
        with torch.no_grad():
            metrics = model.val(
                data=data_yaml,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
                plots=False,
            )

        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        results_table.append((conf, iou, map50, map5095))

        marker = " <-- best" if map50 > best_map50 else ""
        print(f"{conf:>6.2f} {iou:>6.2f} {map50:>8.4f} {map5095:>10.4f}{marker}")

        if map50 > best_map50:
            best_map50 = map50
            best_conf = conf
            best_iou = iou

    print()
    print("=" * 34)
    print(f"Best: conf={best_conf}, iou={best_iou}, mAP50={best_map50:.4f}")
    print()
    print("To apply, update src/constants.py:")
    print(f"  CONFIDENCE_THRESHOLD = {best_conf}")
    print(f"  IOU_THRESHOLD = {best_iou}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep")
    parser.add_argument("--weights", type=str, default="weights/model.pt")
    parser.add_argument("--data", type=str, default="training/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    t_start = time.perf_counter()
    sweep(args.weights, args.data, args.imgsz)
    elapsed = time.perf_counter() - t_start
    print(f"\nSweep completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
