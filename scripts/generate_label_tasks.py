"""Generate label review tasks as JSON for the labeling frontend.

Compares model predictions against ground truth annotations and outputs
flagged disagreements as a JSON file that the labeling frontend can consume.

Usage:
    python scripts/generate_label_tasks.py [--weights weights/model.pt] \
        [--max-images 0] [--conf 0.25]

Output: docs/label_tasks.json
This is a training prep script — can use any imports.
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
TRAIN_IMAGES = REPO / "training" / "data" / "yolo" / "train" / "images"
TRAIN_LABELS = REPO / "training" / "data" / "yolo" / "train" / "labels"
CATEGORY_FILE = REPO / "training" / "data" / "yolo" / "category_names.json"
PRODUCT_IMAGES_DIR = REPO / "training" / "data" / "images"
OUTPUT_JSON = REPO / "docs" / "label_tasks.json"


def load_category_names() -> dict[int, str]:
    raw = json.loads(CATEGORY_FILE.read_text())
    return {int(k): v for k, v in raw.items()}


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append({"class_id": cls, "bbox": [x1, y1, x2, y2]})
    return boxes


def compute_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_to_base64(img: Image.Image, bbox: list[float], pad: float = 0.1) -> str:
    """Crop bbox from image with padding and return as base64 JPEG."""
    img_w, img_h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    # Add padding
    x1 = max(0, x1 - bw * pad)
    y1 = max(0, y1 - bh * pad)
    x2 = min(img_w, x2 + bw * pad)
    y2 = min(img_h, y2 + bh * pad)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    # Resize to max 300px on longest side
    crop.thumbnail((300, 300), Image.LANCZOS)
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="Generate label tasks for review frontend")
    parser.add_argument("--weights", default=str(REPO / "weights" / "model.pt"))
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--flag-conf", type=float, default=0.5)
    args = parser.parse_args()

    cat_names = load_category_names()

    # PyTorch 2.6 compat
    _orig_load = torch.load

    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _patched_load

    print(f"Loading model from {args.weights} ...")
    model = YOLO(args.weights)

    all_images = sorted(
        p
        for p in TRAIN_IMAGES.iterdir()
        if p.stem.startswith("img_") and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.max_images > 0:
        all_images = all_images[: args.max_images]
    print(f"Processing {len(all_images)} training images ...")

    tasks = []

    for idx, img_path in enumerate(all_images):
        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(all_images)}] {img_path.name}")

        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        result = results[0]
        img_h, img_w = result.orig_shape

        preds = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                preds.append(
                    {
                        "class_id": int(box.cls[0].item()),
                        "conf": float(box.conf[0].item()),
                        "bbox": xyxy,
                    }
                )

        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        gt_boxes = load_gt_labels(label_path, img_w, img_h)

        # Open image for cropping
        img = Image.open(str(img_path)).convert("RGB")
        matched_preds = set()

        for gt in gt_boxes:
            best_iou, best_pred = 0.0, None
            best_pred_idx = -1
            for pi, pred in enumerate(preds):
                iou = compute_iou(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
                    best_pred_idx = pi

            if best_pred is not None and best_iou > 0.3:
                matched_preds.add(best_pred_idx)
                if best_pred["class_id"] != gt["class_id"] and best_pred["conf"] >= args.flag_conf:
                    tasks.append(
                        {
                            "id": len(tasks),
                            "type": "wrong_class",
                            "file": img_path.name,
                            "gt_class": gt["class_id"],
                            "gt_name": cat_names.get(gt["class_id"], f"ID {gt['class_id']}"),
                            "pred_class": best_pred["class_id"],
                            "pred_name": cat_names.get(
                                best_pred["class_id"],
                                f"ID {best_pred['class_id']}",
                            ),
                            "conf": round(best_pred["conf"], 3),
                            "iou": round(best_iou, 2),
                            "gt_bbox": [round(v, 1) for v in gt["bbox"]],
                            "pred_bbox": [round(v, 1) for v in best_pred["bbox"]],
                            "crop": crop_to_base64(img, gt["bbox"]),
                        }
                    )

        for pi, pred in enumerate(preds):
            if pi not in matched_preds and pred["conf"] >= args.flag_conf:
                tasks.append(
                    {
                        "id": len(tasks),
                        "type": "missing_annotation",
                        "file": img_path.name,
                        "gt_class": None,
                        "gt_name": None,
                        "pred_class": pred["class_id"],
                        "pred_name": cat_names.get(pred["class_id"], f"ID {pred['class_id']}"),
                        "conf": round(pred["conf"], 3),
                        "iou": 0.0,
                        "gt_bbox": None,
                        "pred_bbox": [round(v, 1) for v in pred["bbox"]],
                        "crop": crop_to_base64(img, pred["bbox"]),
                    }
                )

        img.close()

    # Sort: wrong_class first, then by confidence desc
    type_order = {"wrong_class": 0, "missing_annotation": 1}
    tasks.sort(key=lambda t: (type_order.get(t["type"], 9), -t["conf"]))
    for i, t in enumerate(tasks):
        t["id"] = i

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"categories": cat_names, "tasks": tasks}, ensure_ascii=False)
    OUTPUT_JSON.write_text(data)
    print(f"\nGenerated {len(tasks)} tasks → {OUTPUT_JSON}")
    print(f"  wrong_class:        {sum(1 for t in tasks if t['type'] == 'wrong_class')}")
    print(f"  missing_annotation: {sum(1 for t in tasks if t['type'] == 'missing_annotation')}")


if __name__ == "__main__":
    main()
