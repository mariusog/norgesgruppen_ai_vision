"""Find potential label errors by comparing model predictions against ground truth.

Usage:
    python scripts/review_labels.py [--weights weights/model.pt] [--max-images 50]

Generates docs/label_review.html with flagged disagreements sorted by confidence.
This is a training prep script -- can use any imports.
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

# --- Paths ---
REPO = Path(__file__).resolve().parent.parent
TRAIN_IMAGES = REPO / "training" / "data" / "yolo" / "train" / "images"
TRAIN_LABELS = REPO / "training" / "data" / "yolo" / "train" / "labels"
CATEGORY_FILE = REPO / "training" / "data" / "yolo" / "category_names.json"
OUTPUT_HTML = REPO / "docs" / "label_review.html"


def load_category_names() -> dict[int, str]:
    raw = json.loads(CATEGORY_FILE.read_text())
    return {int(k): v for k, v in raw.items()}


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse YOLO label file into list of {class_id, bbox_xyxy}."""
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


def main():
    parser = argparse.ArgumentParser(description="Review training labels for errors")
    parser.add_argument("--weights", default=str(REPO / "weights" / "model.pt"))
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--conf", type=float, default=0.25, help="Model confidence threshold")
    parser.add_argument("--flag-conf", type=float, default=0.5, help="Min conf to flag wrong class")
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

    # Collect training images (img_* only, skip ref_*)
    all_images = sorted(
        p
        for p in TRAIN_IMAGES.iterdir()
        if p.stem.startswith("img_") and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.max_images > 0:
        all_images = all_images[: args.max_images]
    print(f"Processing {len(all_images)} training images ...")

    flags = []  # list of dicts for the report

    for idx, img_path in enumerate(all_images):
        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(all_images)}] {img_path.name}")

        # Run inference
        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        result = results[0]
        img_h, img_w = result.orig_shape

        # Parse predictions
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

        # Load ground truth
        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        gt_boxes = load_gt_labels(label_path, img_w, img_h)

        matched_preds = set()

        # Check each GT annotation
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
                    flags.append(
                        {
                            "type": "wrong_class",
                            "file": img_path.name,
                            "gt_class": gt["class_id"],
                            "pred_class": best_pred["class_id"],
                            "conf": best_pred["conf"],
                            "iou": best_iou,
                            "gt_bbox": gt["bbox"],
                            "pred_bbox": best_pred["bbox"],
                        }
                    )
            elif best_iou <= 0.3:
                flags.append(
                    {
                        "type": "undetected",
                        "file": img_path.name,
                        "gt_class": gt["class_id"],
                        "pred_class": None,
                        "conf": 0.0,
                        "iou": best_iou,
                        "gt_bbox": gt["bbox"],
                        "pred_bbox": None,
                    }
                )

        # Check unmatched predictions (possible missing annotations)
        for pi, pred in enumerate(preds):
            if pi not in matched_preds and pred["conf"] >= args.flag_conf:
                flags.append(
                    {
                        "type": "missing_annotation",
                        "file": img_path.name,
                        "gt_class": None,
                        "pred_class": pred["class_id"],
                        "conf": pred["conf"],
                        "iou": 0.0,
                        "gt_bbox": None,
                        "pred_bbox": pred["bbox"],
                    }
                )

    # Sort: wrong_class first (most actionable), then by confidence desc
    type_order = {"wrong_class": 0, "missing_annotation": 1, "undetected": 2}
    flags.sort(key=lambda f: (type_order.get(f["type"], 9), -f["conf"]))

    print(f"\nFound {len(flags)} flags:")
    print(f"  wrong_class:        {sum(1 for f in flags if f['type'] == 'wrong_class')}")
    print(f"  missing_annotation: {sum(1 for f in flags if f['type'] == 'missing_annotation')}")
    print(f"  undetected:         {sum(1 for f in flags if f['type'] == 'undetected')}")

    # Generate HTML report
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html(flags, cat_names)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport written to {OUTPUT_HTML}")


def _fmt_bbox(bbox: list[float] | None) -> str:
    if bbox is None:
        return "N/A"
    return f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"


def _build_html(flags: list[dict], cat_names: dict[int, str]) -> str:
    type_colors = {
        "wrong_class": "#ffcccc",
        "missing_annotation": "#ffffcc",
        "undetected": "#ffeecc",
    }
    type_labels = {
        "wrong_class": "WRONG CLASS",
        "missing_annotation": "MISSING ANNOTATION",
        "undetected": "UNDETECTED GT",
    }

    rows = []
    for i, f in enumerate(flags):
        bg = type_colors.get(f["type"], "#ffffff")
        gt_name = cat_names.get(f["gt_class"], "N/A") if f["gt_class"] is not None else "N/A"
        pred_name = cat_names.get(f["pred_class"], "N/A") if f["pred_class"] is not None else "N/A"
        rows.append(f"""<tr style="background:{bg}">
<td><input type="checkbox" id="cb{i}"></td>
<td>{type_labels.get(f["type"], f["type"])}</td>
<td>{f["file"]}</td>
<td title="ID {f["gt_class"]}">{gt_name}</td>
<td title="ID {f["pred_class"]}">{pred_name}</td>
<td>{f["conf"]:.3f}</td>
<td>{f["iou"]:.2f}</td>
<td>{_fmt_bbox(f["gt_bbox"])}</td>
<td>{_fmt_bbox(f["pred_bbox"])}</td>
</tr>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Label Review Report</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
h1 {{ color: #333; }}
.stats {{ margin: 10px 0 20px; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #444; color: #fff; position: sticky; top: 0; }}
tr:hover {{ outline: 2px solid #007bff; }}
.legend {{ display: flex; gap: 20px; margin: 10px 0; }}
.legend span {{ padding: 4px 12px; border-radius: 3px; font-size: 13px; }}
</style></head><body>
<h1>Label Review Report</h1>
<div class="stats">
<b>Total flags:</b> {len(flags)} |
<b>Wrong class:</b> {sum(1 for f in flags if f["type"] == "wrong_class")} |
<b>Missing annotation:</b> {sum(1 for f in flags if f["type"] == "missing_annotation")} |
<b>Undetected GT:</b> {sum(1 for f in flags if f["type"] == "undetected")}
</div>
<div class="legend">
<span style="background:#ffcccc">Wrong class (likely error)</span>
<span style="background:#ffffcc">Missing annotation</span>
<span style="background:#ffeecc">Undetected GT</span>
</div>
<table>
<thead><tr>
<th>Fix</th><th>Type</th><th>Image</th><th>GT Label</th><th>Predicted Label</th>
<th>Confidence</th><th>IoU</th><th>GT BBox (xyxy)</th><th>Pred BBox (xyxy)</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
<script>
document.querySelectorAll('input[type=checkbox]').forEach(cb => {{
  cb.addEventListener('change', () => {{
    const count = document.querySelectorAll('input[type=checkbox]:checked').length;
    document.title = 'Label Review (' + count + ' selected)';
  }});
}});
</script>
</body></html>"""


if __name__ == "__main__":
    main()
