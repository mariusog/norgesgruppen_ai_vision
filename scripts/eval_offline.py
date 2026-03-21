"""Offline score estimation using validation set.

Simulates the competition scoring formula:
  Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

Uses ultralytics built-in validation which computes both metrics.

Usage:
    python scripts/eval_offline.py --weights weights/model.pt
    python scripts/eval_offline.py --weights weights/model.pt --imgsz 1280
    python scripts/eval_offline.py --weights weights/model.pt --conf 0.15 --iou 0.5
    python scripts/eval_offline.py --weights weights/model.pt --tta
    python scripts/eval_offline.py --compare weights/v1.pt weights/v2.pt
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import time
from pathlib import Path
from typing import Any

import torch

torch.load = functools.partial(torch.load, weights_only=False)

from ultralytics import YOLO  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "eval_results.json"


def _load_class_names(data_yaml: str) -> dict[int, str]:
    """Load class names from the YOLO data.yaml file."""
    import yaml

    path = Path(data_yaml)
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    return {int(k): v for k, v in names.items()}


def evaluate(
    weights: str,
    data: str,
    imgsz: int,
    conf: float,
    iou: float,
    tta: bool = False,
) -> dict[str, Any]:
    """Run validation and compute estimated competition score.

    Returns a dict with all metrics for downstream use.
    """
    model = YOLO(weights)

    mode_label = "TTA" if tta else "standard"
    print(f"Evaluating: {weights} ({mode_label})")
    print(f"  imgsz={imgsz}, conf={conf}, iou={iou}, augment={tta}")
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
            augment=tta,
        )
    elapsed = time.perf_counter() - t_start

    # Extract per-class AP50 values for category breakdown
    # metrics.box.ap50 is a numpy array of shape (num_classes,)
    per_class_ap50 = None
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        per_class_ap50 = metrics.box.ap50

    cls_map50 = float(metrics.box.map50)
    cls_map5095 = float(metrics.box.map)

    # Detection mAP (category-agnostic) via single_cls=True
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
            augment=tta,
        )

    det_map50 = float(det_metrics.box.map50)

    # Competition score formula
    score = 0.7 * det_map50 + 0.3 * cls_map50

    print()
    print("=" * 60)
    print(f"  Weights:                     {weights}")
    print(f"  Mode:                        {mode_label}")
    print(f"  Detection mAP@0.5:          {det_map50:.4f}")
    print(f"  Classification mAP@0.5:     {cls_map50:.4f}")
    print(f"  Classification mAP@0.5-0.95: {cls_map5095:.4f}")
    print()
    print("  Estimated competition score:")
    print(f"    0.7 x {det_map50:.4f} + 0.3 x {cls_map50:.4f} = {score:.4f}")
    print("=" * 60)
    print(f"  Evaluation time: {elapsed:.1f}s")

    # --- Per-category breakdown: top-10 worst categories ---
    class_names = _load_class_names(data)
    worst_categories: list[dict[str, Any]] = []
    if per_class_ap50 is not None and len(per_class_ap50) > 0:
        # Build list of (class_id, ap50, name)
        cat_results = []
        for idx in range(len(per_class_ap50)):
            ap = float(per_class_ap50[idx])
            name = class_names.get(idx, f"class_{idx}")
            cat_results.append({"class_id": idx, "name": name, "ap50": ap})

        # Sort by AP50 ascending (worst first)
        cat_results.sort(key=lambda x: x["ap50"])
        worst_10 = cat_results[:10]

        print()
        print("--- Top-10 worst-performing categories (lowest AP@0.5) ---")
        print(f"  {'ID':>4s}  {'AP@0.5':>7s}  Name")
        print(f"  {'----':>4s}  {'------':>7s}  ----")
        for cat in worst_10:
            print(f"  {cat['class_id']:4d}  {cat['ap50']:7.4f}  {cat['name']}")

        worst_categories = worst_10

    print()
    print(
        "NOTE: This is on the validation set. "
        "Competition uses a different test set -- actual score will vary."
    )

    return {
        "weights": weights,
        "mode": mode_label,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "det_map50": det_map50,
        "cls_map50": cls_map50,
        "cls_map5095": cls_map5095,
        "score": score,
        "elapsed_s": round(elapsed, 1),
        "worst_categories": worst_categories,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }


def _save_results(results: list[dict[str, Any]]) -> None:
    """Append results to docs/eval_results.json."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.extend(results)

    with open(RESULTS_PATH, "w") as f:
        f.write(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"\nResults saved to {RESULTS_PATH}")


def compare(
    weights_list: list[str],
    data: str,
    imgsz: int,
    conf: float,
    iou: float,
    tta: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate multiple models and show side-by-side comparison."""
    all_results: list[dict[str, Any]] = []
    for w in weights_list:
        print(f"\n{'#' * 60}")
        print(f"# Evaluating: {w}")
        print(f"{'#' * 60}\n")
        result = evaluate(w, data, imgsz, conf, iou, tta=tta)
        all_results.append(result)

    # Print comparison table
    print()
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Header
    col_w = max(len(Path(w).name) for w in weights_list) + 2
    col_w = max(col_w, 12)
    header = (
        f"  {'Model':<{col_w}s}"
        f"  {'Det mAP50':>10s}"
        f"  {'Cls mAP50':>10s}"
        f"  {'Score':>8s}"
        f"  {'Time (s)':>9s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_results:
        name = Path(r["weights"]).name
        line = (
            f"  {name:<{col_w}s}"
            f"  {r['det_map50']:>10.4f}"
            f"  {r['cls_map50']:>10.4f}"
            f"  {r['score']:>8.4f}"
            f"  {r['elapsed_s']:>9.1f}"
        )
        print(line)

    # Delta row if exactly 2 models
    if len(all_results) == 2:
        a, b = all_results
        print("  " + "-" * (len(header) - 2))
        delta_det = b["det_map50"] - a["det_map50"]
        delta_cls = b["cls_map50"] - a["cls_map50"]
        delta_score = b["score"] - a["score"]
        delta_time = b["elapsed_s"] - a["elapsed_s"]

        def sign(v):
            return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

        def sign_t(v):
            return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

        line = (
            f"  {'Delta':<{col_w}s}"
            f"  {sign(delta_det):>10s}"
            f"  {sign(delta_cls):>10s}"
            f"  {sign(delta_score):>8s}"
            f"  {sign_t(delta_time):>9s}"
        )
        print(line)

    print("=" * 80)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline score estimation")
    parser.add_argument("--weights", type=str, default="weights/model.pt")
    parser.add_argument("--data", type=str, default="training/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (augment=True)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="WEIGHTS",
        help="Compare multiple weight files side-by-side (e.g. --compare w1.pt w2.pt)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to docs/eval_results.json",
    )
    args = parser.parse_args()

    if args.compare:
        results = compare(args.compare, args.data, args.imgsz, args.conf, args.iou, tta=args.tta)
    else:
        result = evaluate(args.weights, args.data, args.imgsz, args.conf, args.iou, tta=args.tta)
        results = [result]

    if not args.no_save:
        _save_results(results)


if __name__ == "__main__":
    main()
