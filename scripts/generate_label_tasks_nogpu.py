"""Generate label review tasks from COCO annotations WITHOUT model inference.

Flags potential annotation issues by analyzing ground truth statistics:
  - Annotations with very small bounding boxes (<20x20 pixels)
  - Images with very few annotations (<3)
  - Categories that appear very rarely (<3 occurrences)

Usage:
    python scripts/generate_label_tasks_nogpu.py [--min-box-size 20] [--min-ann-per-image 3] [--min-cat-count 3]

Output: docs/label_tasks.json (same format as generate_label_tasks.py)
"""

import argparse
import base64
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = REPO / "training" / "data" / "train" / "annotations.json"
TRAIN_IMAGES_DIR = REPO / "training" / "data" / "train" / "images"
CATEGORY_FILE = REPO / "training" / "data" / "yolo" / "category_names.json"
OUTPUT_JSON = REPO / "docs" / "label_tasks.json"


def load_category_names() -> dict[int, str]:
    raw = json.loads(CATEGORY_FILE.read_text())
    return {int(k): v for k, v in raw.items()}


def crop_to_base64(img: Image.Image, bbox_xywh: list[float], pad: float = 0.1) -> str:
    """Crop bbox (COCO xywh format) from image with padding; return base64 JPEG."""
    img_w, img_h = img.size
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    bw, bh = w, h
    # Use much more padding for tiny boxes so context is visible
    min_crop = 150  # minimum crop dimension in pixels
    if bw < min_crop or bh < min_crop:
        pad_x = max(pad, (min_crop - bw) / (2 * max(bw, 1)))
        pad_y = max(pad, (min_crop - bh) / (2 * max(bh, 1)))
    else:
        pad_x, pad_y = pad, pad
    # Add padding
    x1 = max(0, x1 - bw * pad_x)
    y1 = max(0, y1 - bh * pad_y)
    x2 = min(img_w, x2 + bw * pad_x)
    y2 = min(img_h, y2 + bh * pad_y)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    # Resize to max 300px on longest side
    crop.thumbnail((300, 300), Image.LANCZOS)
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="Generate label tasks without GPU")
    parser.add_argument("--min-box-size", type=int, default=20,
                        help="Flag boxes smaller than NxN pixels (default: 20)")
    parser.add_argument("--min-ann-per-image", type=int, default=3,
                        help="Flag images with fewer than N annotations (default: 3)")
    parser.add_argument("--min-cat-count", type=int, default=3,
                        help="Flag categories with fewer than N occurrences (default: 3)")
    args = parser.parse_args()

    # Load data
    coco = json.loads(ANNOTATIONS_FILE.read_text())
    cat_names = load_category_names()

    images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    cat_counter: Counter[int] = Counter()

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)
        cat_counter[ann["category_id"]] += 1

    # Identify rare categories
    rare_cats = {cat_id for cat_id, count in cat_counter.items()
                 if count < args.min_cat_count}

    # Identify images with few annotations
    sparse_images = {img_id for img_id, anns in anns_by_image.items()
                     if len(anns) < args.min_ann_per_image}

    print(f"Dataset: {len(images_by_id)} images, {len(coco['annotations'])} annotations")
    print(f"Rare categories (<{args.min_cat_count} occurrences): {len(rare_cats)}")
    print(f"Sparse images (<{args.min_ann_per_image} annotations): {len(sparse_images)}")

    tasks: list[dict] = []
    # Track images we need to open - group annotations by image for efficiency
    image_tasks: dict[int, list[dict]] = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img_info = images_by_id[img_id]
        bbox = ann["bbox"]  # COCO format: [x, y, width, height]
        bw, bh = bbox[2], bbox[3]
        cat_id = ann["category_id"]

        reasons = []
        # Priority scoring: lower = more suspicious
        priority = 100

        # Check small box
        if bw < args.min_box_size or bh < args.min_box_size:
            reasons.append(f"tiny_box ({int(bw)}x{int(bh)}px)")
            priority -= 40

        # Check rare category
        if cat_id in rare_cats:
            count = cat_counter[cat_id]
            reasons.append(f"rare_category (count={count})")
            priority -= 30

        # Check sparse image
        if img_id in sparse_images:
            num_anns = len(anns_by_image[img_id])
            reasons.append(f"sparse_image ({num_anns} annotations)")
            priority -= 20

        if not reasons:
            continue

        # Convert bbox to xyxy for the task format (matches generate_label_tasks.py)
        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]

        task = {
            "id": 0,  # assigned later
            "type": "suspicious_annotation",
            "file": img_info["file_name"],
            "gt_class": cat_id,
            "gt_name": cat_names.get(cat_id, f"ID {cat_id}"),
            "pred_class": None,
            "pred_name": None,
            "conf": 0.0,
            "iou": 0.0,
            "gt_bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "pred_bbox": None,
            "reasons": reasons,
            "priority": priority,
            "crop": None,  # filled in later
            "_img_id": img_id,
            "_bbox_xywh": bbox,
        }
        image_tasks.setdefault(img_id, []).append(task)

    # Now open each image once and crop all flagged annotations
    total_images = len(image_tasks)
    for idx, (img_id, img_task_list) in enumerate(image_tasks.items()):
        img_info = images_by_id[img_id]
        img_path = TRAIN_IMAGES_DIR / img_info["file_name"]

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Cropping [{idx+1}/{total_images}] {img_info['file_name']} "
                  f"({len(img_task_list)} flagged annotations)")

        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping crops")
            for t in img_task_list:
                t["crop"] = ""
                tasks.append(t)
            continue

        img = Image.open(str(img_path)).convert("RGB")
        for t in img_task_list:
            t["crop"] = crop_to_base64(img, t["_bbox_xywh"])
            tasks.append(t)
        img.close()

    # Sort by priority (most suspicious first), then by file name
    tasks.sort(key=lambda t: (t["priority"], t["file"]))

    # Clean up internal fields and assign IDs
    for i, t in enumerate(tasks):
        t["id"] = i
        del t["_img_id"]
        del t["_bbox_xywh"]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"categories": {str(k): v for k, v in cat_names.items()}, "tasks": tasks}
    OUTPUT_JSON.write_text(json.dumps(output_data, ensure_ascii=False))

    print(f"\nGenerated {len(tasks)} tasks -> {OUTPUT_JSON}")
    reason_counts: Counter[str] = Counter()
    for t in tasks:
        for r in t["reasons"]:
            tag = r.split(" ")[0]
            reason_counts[tag] += 1
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
