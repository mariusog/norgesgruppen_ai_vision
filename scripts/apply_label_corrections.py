"""Apply label corrections from the labeling frontend back to YOLO label files.

Usage:
    python scripts/apply_label_corrections.py label_corrections.json [--dry-run]

Reads the corrections JSON exported from docs/labeler.html and modifies
the YOLO .txt label files accordingly.

This is a training prep script — can use any imports.
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_LABELS = REPO / "training" / "data" / "yolo" / "train" / "labels"
TRAIN_IMAGES = REPO / "training" / "data" / "yolo" / "train" / "images"


def get_image_size(img_name: str) -> tuple[int, int]:
    """Return (width, height) of the training image."""
    from PIL import Image

    img_path = TRAIN_IMAGES / img_name
    with Image.open(str(img_path)) as img:
        return img.size


def xyxy_to_yolo(
    bbox_xyxy: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] pixel coords to YOLO [cx, cy, w, h] normalized."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def main():
    parser = argparse.ArgumentParser(description="Apply label corrections to YOLO files")
    parser.add_argument("corrections_file", type=str, help="Path to label_corrections.json")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    args = parser.parse_args()

    corrections = json.loads(Path(args.corrections_file).read_text())
    print(f"Loaded {len(corrections)} corrections")

    # Group by file
    by_file: dict[str, list] = {}
    for c in corrections:
        by_file.setdefault(c["file"], []).append(c)

    modified = 0
    for file_name, file_corrections in sorted(by_file.items()):
        label_file = TRAIN_LABELS / (Path(file_name).stem + ".txt")

        if not label_file.exists():
            print(f"  SKIP {label_file} (not found)")
            continue

        img_w, img_h = get_image_size(file_name)
        lines = label_file.read_text().strip().splitlines()
        new_lines = list(lines)  # copy
        additions = []

        for corr in file_corrections:
            if corr["decision"] == "use_pred" and corr.get("bbox"):
                # Change class of the matching GT line
                bbox = corr["bbox"]
                cx_t, cy_t, w_t, h_t = xyxy_to_yolo(bbox, img_w, img_h)
                best_idx, best_dist = -1, 999.0
                for i, line in enumerate(new_lines):
                    parts = line.split()
                    cx, cy = float(parts[1]), float(parts[2])
                    dist = abs(cx - cx_t) + abs(cy - cy_t)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0 and best_dist < 0.1:
                    parts = new_lines[best_idx].split()
                    old_cls = parts[0]
                    parts[0] = str(corr["newClass"])
                    new_lines[best_idx] = " ".join(parts)
                    print(f"  {file_name}: class {old_cls} → {corr['newClass']} (line {best_idx})")

            elif (
                corr["decision"] == "other"
                and corr.get("bbox")
                and corr.get("newClass") is not None
            ):
                bbox = corr["bbox"]
                cx_t, cy_t, w_t, h_t = xyxy_to_yolo(bbox, img_w, img_h)
                # Try to find and update existing line
                best_idx, best_dist = -1, 999.0
                for i, line in enumerate(new_lines):
                    parts = line.split()
                    cx, cy = float(parts[1]), float(parts[2])
                    dist = abs(cx - cx_t) + abs(cy - cy_t)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0 and best_dist < 0.1:
                    parts = new_lines[best_idx].split()
                    old_cls = parts[0]
                    parts[0] = str(corr["newClass"])
                    new_lines[best_idx] = " ".join(parts)
                    print(f"  {file_name}: class {old_cls} → {corr['newClass']} (line {best_idx})")
                else:
                    # Add new annotation
                    new_line = f"{corr['newClass']} {cx_t:.6f} {cy_t:.6f} {w_t:.6f} {h_t:.6f}"
                    additions.append(new_line)
                    print(f"  {file_name}: ADD class {corr['newClass']}")

            elif corr["decision"] == "confirm_pred" and corr.get("type") == "missing_annotation":
                # Add the predicted bbox as a new annotation
                bbox = corr.get("bbox") or corr.get("pred_bbox")
                if bbox:
                    cx, cy, w, h = xyxy_to_yolo(bbox, img_w, img_h)
                    new_line = f"{corr['pred_class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    additions.append(new_line)
                    print(f"  {file_name}: ADD class {corr.get('pred_class', '?')}")

            elif corr["decision"] == "mark_unknown" and corr.get("bbox"):
                bbox = corr["bbox"]
                cx_t, cy_t, w_t, h_t = xyxy_to_yolo(bbox, img_w, img_h)
                best_idx, best_dist = -1, 999.0
                for i, line in enumerate(new_lines):
                    parts = line.split()
                    cx, cy = float(parts[1]), float(parts[2])
                    dist = abs(cx - cx_t) + abs(cy - cy_t)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0 and best_dist < 0.1:
                    parts = new_lines[best_idx].split()
                    old_cls = parts[0]
                    parts[0] = "355"
                    new_lines[best_idx] = " ".join(parts)
                    print(f"  {file_name}: class {old_cls} → 355 (unknown)")

        new_lines.extend(additions)

        if new_lines != list(lines) or additions:
            modified += 1
            if not args.dry_run:
                # Backup
                backup = label_file.with_suffix(".txt.bak")
                if not backup.exists():
                    shutil.copy2(label_file, backup)
                label_file.write_text("\n".join(new_lines) + "\n")

    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {modified} label files")


if __name__ == "__main__":
    main()
