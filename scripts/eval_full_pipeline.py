"""Full-pipeline offline evaluation on the validation set.

Runs the EXACT same inference pipeline as run.py (ensemble + classifier + TTA)
on the validation set, then computes detection and classification mAP@0.5 using
pycocotools to simulate competition scoring.

Estimated competition score: 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

This is a TRAINING/EVAL script -- it is NOT submission code, so it can use os, sys, etc.

Usage:
    # Default (uses constants from src/constants.py):
    python scripts/eval_full_pipeline.py

    # Override ensemble weights:
    python scripts/eval_full_pipeline.py \
        --ensemble-weights weights/yolov8l-1280-aug.pt,weights/yolov8l-640-aug.pt

    # Override classifier:
    python scripts/eval_full_pipeline.py \
        --classifier weights/classifier_v2.pt \
        --classifier-model convnext_small.fb_in22k_ft_in1k

    # Disable TTA for faster eval:
    python scripts/eval_full_pipeline.py --no-tta --no-classifier-tta

    # Full custom config:
    python scripts/eval_full_pipeline.py \
        --ensemble-weights weights/a.pt,weights/b.pt \
        --ensemble-sizes 1280,640 \
        --classifier weights/cls.pt \
        --conf 0.01 --iou 0.45 --wbf-iou 0.55 \
        --no-tta
"""

from __future__ import annotations

import argparse
import copy
import datetime
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6 patch — must come before ultralytics import
torch.load = functools.partial(torch.load, weights_only=False)

# ---------------------------------------------------------------------------
# GCS dataset download (reused from training/train.py pattern)
# ---------------------------------------------------------------------------

GCS_BUCKET = "ai-nm26osl-1792-nmiai"


def download_dataset_from_gcs() -> None:
    """Pull dataset from GCS if running on Vertex AI."""
    workspace = Path("/workspace")
    if not workspace.exists():
        print("Not running in Vertex AI container — skipping GCS download.")
        return
    data_dir = workspace / "data"
    if data_dir.exists() and any(data_dir.rglob("*.jpg")):
        print(f"Dataset already present at {data_dir}")
        return
    print(f"Pulling dataset from gs://{GCS_BUCKET}/datasets/yolo/ -> {data_dir}")
    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix="datasets/yolo/"))
    data_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        dest = data_dir / Path(blob.name).relative_to("datasets/yolo")
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        count += 1
    print(f"Downloaded {count} files to {data_dir}")

    # Also download COCO annotations if they exist in GCS
    anno_blob_name = "datasets/train/annotations.json"
    anno_dest = data_dir / "train" / "annotations.json"
    if not anno_dest.exists():
        try:
            blob = bucket.blob(anno_blob_name)
            if blob.exists():
                anno_dest.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(anno_dest))
                print(f"Downloaded annotations to {anno_dest}")
        except Exception as e:
            print(f"Warning: could not download annotations from GCS: {e}")


def download_weights_from_gcs(weight_paths: list[str]) -> None:
    """Download any missing weight files from GCS."""
    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for wp in weight_paths:
        p = Path(wp)
        if p.exists():
            continue
        gcs_key = f"weights/{p.name}"
        print(f"Downloading gs://{GCS_BUCKET}/{gcs_key} -> {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = bucket.blob(gcs_key)
            blob.download_to_filename(str(p))
            print(f"  Downloaded ({p.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  WARNING: failed to download {gcs_key}: {e}")


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def find_annotations_file() -> Path:
    """Locate the COCO-format annotations.json for the validation set."""
    candidates = [
        Path("/workspace/data/val/annotations.json"),
        Path("/workspace/data/train/annotations.json"),
        Path("training/data/val/annotations.json"),
        Path("training/data/train/annotations.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find annotations.json. Searched:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def find_val_images_dir() -> Path:
    """Locate the validation images directory."""
    candidates = [
        Path("/workspace/data/val/images"),
        Path("training/data/val/images"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Cannot find val/images directory. Searched:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def load_coco_annotations(ann_path: Path) -> dict:
    """Load COCO-format annotations and build lookup structures."""
    with open(ann_path) as f:
        coco_data = json.load(f)

    # Build image_id -> filename mapping
    id_to_filename: dict[int, str] = {}
    for img in coco_data.get("images", []):
        id_to_filename[img["id"]] = img["file_name"]

    return coco_data


# ---------------------------------------------------------------------------
# mAP computation using pycocotools
# ---------------------------------------------------------------------------


def compute_maps(
    gt_annotations: dict,
    predictions: list[dict],
    val_image_ids: set[int],
) -> dict[str, float]:
    """Compute detection mAP@0.5 and classification mAP@0.5 using pycocotools.

    Args:
        gt_annotations: Full COCO-format annotation dict.
        predictions: List of prediction dicts from our pipeline.
        val_image_ids: Set of image IDs that are in the validation set.

    Returns:
        Dict with det_map50, cls_map50, and estimated competition score.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Filter GT to only include val images
    gt_filtered = copy.deepcopy(gt_annotations)
    gt_filtered["images"] = [img for img in gt_filtered["images"] if img["id"] in val_image_ids]
    gt_filtered["annotations"] = [
        ann for ann in gt_filtered["annotations"] if ann["image_id"] in val_image_ids
    ]

    # Filter predictions to only include val images
    pred_filtered = [p for p in predictions if p["image_id"] in val_image_ids]

    if not pred_filtered:
        print("WARNING: No predictions for validation images!")
        return {"det_map50": 0.0, "cls_map50": 0.0, "score": 0.0}

    # --- Classification mAP@0.5 (standard: category must match) ---
    print("\n--- Computing classification mAP@0.5 ---")
    coco_gt = COCO()
    coco_gt.dataset = gt_filtered
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(pred_filtered)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])  # Only IoU=0.5
    coco_eval.params.maxDets = [1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    cls_map50 = float(coco_eval.stats[0])  # AP@IoU=0.5

    # --- Detection mAP@0.5 (category-agnostic: all predictions mapped to cat 1) ---
    print("\n--- Computing detection mAP@0.5 (category-agnostic) ---")
    gt_single_cls = copy.deepcopy(gt_filtered)
    for ann in gt_single_cls["annotations"]:
        ann["category_id"] = 1
    gt_single_cls["categories"] = [{"id": 1, "name": "object"}]

    pred_single_cls = copy.deepcopy(pred_filtered)
    for p in pred_single_cls:
        p["category_id"] = 1

    coco_gt_det = COCO()
    coco_gt_det.dataset = gt_single_cls
    coco_gt_det.createIndex()

    coco_dt_det = coco_gt_det.loadRes(pred_single_cls)
    coco_eval_det = COCOeval(coco_gt_det, coco_dt_det, "bbox")
    coco_eval_det.params.iouThrs = np.array([0.5])
    coco_eval_det.params.maxDets = [1000]
    coco_eval_det.evaluate()
    coco_eval_det.accumulate()
    coco_eval_det.summarize()
    det_map50 = float(coco_eval_det.stats[0])

    # Competition score
    score = 0.7 * det_map50 + 0.3 * cls_map50

    return {
        "det_map50": det_map50,
        "cls_map50": cls_map50,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------


def apply_config_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Apply CLI overrides to src/constants module and return a summary dict."""
    import src.constants as C

    overrides: dict[str, object] = {}

    if args.ensemble_weights is not None:
        weights = [w.strip() for w in args.ensemble_weights.split(",") if w.strip()]
        C.ENSEMBLE_WEIGHTS = weights  # type: ignore[assignment]
        overrides["ENSEMBLE_WEIGHTS"] = weights

    if args.ensemble_sizes is not None:
        sizes = [int(s.strip()) for s in args.ensemble_sizes.split(",") if s.strip()]
        C.ENSEMBLE_IMAGE_SIZES = sizes  # type: ignore[assignment]
        overrides["ENSEMBLE_IMAGE_SIZES"] = sizes

    if args.classifier is not None:
        C.CLASSIFIER_PATH = args.classifier  # type: ignore[assignment]
        C.USE_CLASSIFIER = True  # type: ignore[assignment]
        overrides["CLASSIFIER_PATH"] = args.classifier

    if args.classifier_model is not None:
        C.CLASSIFIER_MODEL_NAME = args.classifier_model  # type: ignore[assignment]
        overrides["CLASSIFIER_MODEL_NAME"] = args.classifier_model

    if args.conf is not None:
        C.CONFIDENCE_THRESHOLD = args.conf  # type: ignore[assignment]
        overrides["CONFIDENCE_THRESHOLD"] = args.conf

    if args.iou is not None:
        C.IOU_THRESHOLD = args.iou  # type: ignore[assignment]
        overrides["IOU_THRESHOLD"] = args.iou

    if args.wbf_iou is not None:
        C.WBF_IOU_THRESHOLD = args.wbf_iou  # type: ignore[assignment]
        overrides["WBF_IOU_THRESHOLD"] = args.wbf_iou

    if args.no_tta:
        C.USE_TTA = False  # type: ignore[assignment]
        overrides["USE_TTA"] = False

    if args.no_classifier_tta:
        C.USE_CLASSIFIER_TTA = False  # type: ignore[assignment]
        overrides["USE_CLASSIFIER_TTA"] = False

    if args.no_classifier:
        C.USE_CLASSIFIER = False  # type: ignore[assignment]
        overrides["USE_CLASSIFIER"] = False

    if args.no_ensemble:
        C.ENSEMBLE_WEIGHTS = []  # type: ignore[assignment]
        overrides["ENSEMBLE_WEIGHTS"] = []

    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-pipeline offline evaluation (simulates competition score)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="Comma-separated paths to ensemble weight files (overrides constants)",
    )
    parser.add_argument(
        "--ensemble-sizes",
        type=str,
        default=None,
        help="Comma-separated image sizes per ensemble model (overrides constants)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Path to classifier weights (overrides CLASSIFIER_PATH)",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="timm model name for classifier (overrides CLASSIFIER_MODEL_NAME)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (overrides CONFIDENCE_THRESHOLD)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold (overrides IOU_THRESHOLD)",
    )
    parser.add_argument(
        "--wbf-iou",
        type=float,
        default=None,
        help="WBF IoU threshold (overrides WBF_IOU_THRESHOLD)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation for YOLO",
    )
    parser.add_argument(
        "--no-classifier-tta",
        action="store_true",
        help="Disable classifier TTA",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Disable the two-stage classifier entirely",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble, use single MODEL_PATH instead",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to COCO annotations.json (auto-detected if not specified)",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default=None,
        help="Path to validation images directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save predictions JSON to this path",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Append results to this JSON file (default: docs/eval_pipeline_results.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Full Pipeline Offline Evaluation")
    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"  {now.isoformat()}")
    print("=" * 70)

    # --- Step 1: Download data if on Vertex AI ---
    download_dataset_from_gcs()

    # --- Step 2: Apply config overrides BEFORE importing run.py functions ---
    # We must override src.constants before run.py reads them at import time.
    # Since run.py imports constants at module level, we modify the module attrs
    # before importing run.py functions.

    # First, ensure src is importable
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    overrides = apply_config_overrides(args)

    if overrides:
        print("\nConfig overrides:")
        for k, v in overrides.items():
            print(f"  {k} = {v}")
    else:
        print("\nUsing default config from src/constants.py")

    # Now import run.py functions (they will read the already-modified constants)
    from run import (
        classify_crops,
        collect_images,
        load_classifier,
        load_ensemble_models,
        load_model,
        run_ensemble_inference,
        run_inference,
    )
    from src.constants import ENSEMBLE_WEIGHTS

    # Download weights from GCS if missing (on Vertex AI)
    if Path("/workspace").exists():
        all_weight_paths = list(ENSEMBLE_WEIGHTS)
        from src.constants import CLASSIFIER_PATH, MODEL_PATH, USE_CLASSIFIER

        if not ENSEMBLE_WEIGHTS:
            all_weight_paths.append(MODEL_PATH)
        if USE_CLASSIFIER:
            all_weight_paths.append(CLASSIFIER_PATH)
        download_weights_from_gcs(all_weight_paths)

    # --- Step 3: Find annotations and val images ---
    ann_path = Path(args.annotations) if args.annotations else find_annotations_file()
    print(f"\nAnnotations: {ann_path}")

    val_dir = Path(args.val_images) if args.val_images else find_val_images_dir()
    print(f"Val images:  {val_dir}")

    gt_data = load_coco_annotations(ann_path)

    # Build set of val image IDs from the actual image files
    image_paths = collect_images(val_dir)
    val_image_ids = {int(p.stem.split("_")[-1]) for p in image_paths}
    print(f"Val images found: {len(image_paths)}")

    # Check how many GT annotations match val images
    gt_val_anns = [a for a in gt_data["annotations"] if a["image_id"] in val_image_ids]
    print(f"GT annotations for val: {len(gt_val_anns)}")

    if not gt_val_anns:
        print("\nERROR: No ground truth annotations match the validation image IDs!")
        print("This likely means the annotations.json is for a different split.")
        print("Val image IDs (first 10):", sorted(val_image_ids)[:10])
        gt_img_ids = {a["image_id"] for a in gt_data["annotations"]}
        print("GT image IDs (first 10):", sorted(gt_img_ids)[:10])
        sys.exit(1)

    # --- Step 4: Run full inference pipeline ---
    print("\n" + "=" * 70)
    print("  Running inference pipeline")
    print("=" * 70)

    from src.constants import (
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        USE_CLASSIFIER,
        USE_CLASSIFIER_TTA,
        USE_TTA,
        WBF_IOU_THRESHOLD,
    )

    print(f"  ENSEMBLE_WEIGHTS:    {ENSEMBLE_WEIGHTS}")
    print(f"  CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"  IOU_THRESHOLD:        {IOU_THRESHOLD}")
    print(f"  WBF_IOU_THRESHOLD:    {WBF_IOU_THRESHOLD}")
    print(f"  USE_TTA:              {USE_TTA}")
    print(f"  USE_CLASSIFIER:       {USE_CLASSIFIER}")
    print(f"  USE_CLASSIFIER_TTA:   {USE_CLASSIFIER_TTA}")
    print()

    t_start = time.perf_counter()

    # Load classifier first (before YOLO models to catch errors early)
    classifiers = load_classifier()

    # Run detection
    if ENSEMBLE_WEIGHTS:
        models = load_ensemble_models()
        predictions = run_ensemble_inference(models, image_paths)
    else:
        model = load_model()
        predictions = run_inference(model, image_paths)

    t_detection = time.perf_counter() - t_start
    print(f"Detection complete: {len(predictions)} raw detections in {t_detection:.1f}s")

    # Run classifier
    if classifiers is not None:
        t_cls_start = time.perf_counter()
        predictions = classify_crops(image_paths, predictions, classifiers)
        t_cls = time.perf_counter() - t_cls_start
        print(f"Classification complete in {t_cls:.1f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nTotal inference time: {t_total:.1f}s for {len(image_paths)} images")
    print(f"  ({t_total / max(len(image_paths), 1) * 1000:.0f} ms/image)")

    # --- Step 5: Save predictions if requested ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(predictions, f)
        print(f"\nPredictions saved to {out_path}")

    # --- Step 6: Compute mAP ---
    print("\n" + "=" * 70)
    print("  Computing metrics")
    print("=" * 70)

    metrics = compute_maps(gt_data, predictions, val_image_ids)

    # --- Step 7: Print results ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Detection mAP@0.5:       {metrics['det_map50']:.4f}")
    print(f"  Classification mAP@0.5:  {metrics['cls_map50']:.4f}")
    print()
    print("  Estimated competition score:")
    print(
        f"    0.7 x {metrics['det_map50']:.4f} + "
        f"0.3 x {metrics['cls_map50']:.4f} = {metrics['score']:.4f}"
    )
    print("=" * 70)
    print(f"  Total wall time: {t_total:.1f}s")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Val images:  {len(image_paths)}")
    print()
    print(
        "NOTE: This is on the validation set. "
        "Competition uses a different test set -- actual score will vary."
    )

    # --- Step 8: Save structured results ---
    result_record = {
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "det_map50": metrics["det_map50"],
        "cls_map50": metrics["cls_map50"],
        "score": metrics["score"],
        "num_predictions": len(predictions),
        "num_val_images": len(image_paths),
        "inference_time_s": round(t_total, 1),
        "config": {
            "ensemble_weights": list(ENSEMBLE_WEIGHTS),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "wbf_iou_threshold": WBF_IOU_THRESHOLD,
            "use_tta": USE_TTA,
            "use_classifier": USE_CLASSIFIER,
            "use_classifier_tta": USE_CLASSIFIER_TTA,
        },
        "overrides": {k: str(v) for k, v in overrides.items()},
    }

    results_path = Path(args.results_json or str(repo_root / "docs" / "eval_pipeline_results.json"))
    results_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.loads(f.read())
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(result_record)
    with open(results_path, "w") as f:
        f.write(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()
