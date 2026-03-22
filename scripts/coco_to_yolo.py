"""Convert COCO annotations to YOLO format with train/val split.

Reads annotations.json (COCO format) and produces a YOLO-compatible
directory layout with normalized label files and copied images.

No use of `import os` — pathlib.Path throughout.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# --- Config ---
ANNOTATIONS_PATH = Path("/workspaces/norgesgruppen_ai_vision/training/data/train/annotations.json")
SOURCE_IMAGES_DIR = Path("/workspaces/norgesgruppen_ai_vision/training/data/train/images")
OUTPUT_DIR = Path("/workspaces/norgesgruppen_ai_vision/training/data/yolo")
TRAIN_RATIO = 0.85
RANDOM_SEED = 42


def main() -> None:
    print(f"Reading annotations from {ANNOTATIONS_PATH} ...")
    coco = json.loads(ANNOTATIONS_PATH.read_text())

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    n_img, n_ann, n_cat = len(images), len(annotations), len(categories)
    print(f"  Found {n_img} images, {n_ann} annotations, {n_cat} categories")

    # Build lookup: image_id -> image info
    image_lookup: dict[int, dict] = {img["id"]: img for img in images}

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Train/val split
    image_ids = sorted(image_lookup.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * TRAIN_RATIO)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    print(f"  Split: {len(train_ids)} train, {len(val_ids)} val (seed={RANDOM_SEED})")

    # Create output directories
    splits = {"train": train_ids, "val": val_ids}
    for split_name in splits:
        (OUTPUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # Convert and write
    total_labels = 0
    for split_name, id_set in splits.items():
        print(f"\nProcessing {split_name} split ({len(id_set)} images) ...")
        count = 0
        for img_id in sorted(id_set):
            img_info = image_lookup[img_id]
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            # Copy image
            src = SOURCE_IMAGES_DIR / file_name
            dst = OUTPUT_DIR / split_name / "images" / file_name
            if src.exists():
                shutil.copy2(str(src), str(dst))
            else:
                print(f"  WARNING: source image not found: {src}")
                continue

            # Build YOLO label lines
            lines: list[str] = []
            for ann in anns_by_image.get(img_id, []):
                cat_id = ann["category_id"]
                # COCO bbox is [x_topleft, y_topleft, width, height]
                bx, by, bw, bh = ann["bbox"]
                # Convert to YOLO: x_center, y_center, width, height (normalised 0-1)
                x_center = (bx + bw / 2.0) / img_w
                y_center = (by + bh / 2.0) / img_h
                w_norm = bw / img_w
                h_norm = bh / img_h
                lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            label_path = OUTPUT_DIR / split_name / "labels" / (Path(file_name).stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            total_labels += len(lines)

            count += 1
            if count % 50 == 0:
                print(f"  {count}/{len(id_set)} images done")

        print(f"  Finished {split_name}: {count} images processed")

    print(f"\nTotal label entries written: {total_labels}")

    # Write category_names.json
    cat_map = {cat["id"]: cat["name"] for cat in categories}
    cat_out = OUTPUT_DIR / "category_names.json"
    cat_out.write_text(json.dumps(cat_map, indent=2, ensure_ascii=False))
    print(f"Category mapping written to {cat_out} ({len(cat_map)} categories)")

    print("\nDone! Output directory structure:")
    for split_name in ("train", "val"):
        imgs = list((OUTPUT_DIR / split_name / "images").glob("*.jp*g"))
        lbls = list((OUTPUT_DIR / split_name / "labels").glob("*.txt"))
        print(f"  {split_name}/images: {len(imgs)} files")
        print(f"  {split_name}/labels: {len(lbls)} files")


if __name__ == "__main__":
    main()
