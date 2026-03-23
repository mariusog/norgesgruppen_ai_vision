"""Convert product reference images to YOLO training format and upload to GCS.

For each product with reference images (main.jpg, front.jpg, etc.), creates a
YOLO label with a full-image bounding box (class_id 0.5 0.5 1.0 1.0) and
copies the image + label into the YOLO train split. Then uploads to GCS
using the google-cloud-storage SDK.

This is a training prep script — NOT submission code.
"""

import json
import shutil
from pathlib import Path

from google.cloud import storage

# --- Config ---
METADATA_PATH = Path("./training/data/images/metadata.json")
PRODUCT_IMAGES_DIR = Path("./training/data/images")
CATEGORY_NAMES_PATH = Path(
    "./training/data/yolo/category_names.json"
)
YOLO_TRAIN_IMAGES = Path("./training/data/yolo/train/images")
YOLO_TRAIN_LABELS = Path("./training/data/yolo/train/labels")
GCS_BUCKET = "YOUR_GCS_BUCKET"
GCS_IMAGES_PREFIX = "datasets/yolo/train/images/"
GCS_LABELS_PREFIX = "datasets/yolo/train/labels/"


def upload_to_gcs(local_paths: list[Path], bucket_name: str, prefix: str) -> None:
    """Upload a list of local files to a GCS bucket under the given prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for i, path in enumerate(local_paths, 1):
        blob = bucket.blob(prefix + path.name)
        blob.upload_from_filename(str(path))
        if i % 100 == 0:
            print(f"    Uploaded {i}/{len(local_paths)} ...")
    print(f"    Uploaded {len(local_paths)}/{len(local_paths)} total")


def main() -> None:
    # Load metadata
    print(f"Loading metadata from {METADATA_PATH} ...")
    metadata = json.loads(METADATA_PATH.read_text())
    products = metadata.get("products", metadata)
    if isinstance(products, dict):
        products = list(products.values())
    print(f"  Found {len(products)} products in metadata")

    # Load category name -> id mapping (file is {id_str: name})
    print(f"Loading category names from {CATEGORY_NAMES_PATH} ...")
    cat_id_to_name: dict[str, str] = json.loads(CATEGORY_NAMES_PATH.read_text())
    # Build reverse: product_name -> category_id (int)
    name_to_cat_id: dict[str, int] = {}
    for cat_id_str, name in cat_id_to_name.items():
        name_to_cat_id[name] = int(cat_id_str)
    print(f"  Loaded {len(name_to_cat_id)} category names")

    # Ensure output directories exist
    YOLO_TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    YOLO_TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    # Process products
    images_added = 0
    categories_covered: set[int] = set()
    skipped_no_match = 0
    skipped_no_images = 0
    new_image_paths: list[Path] = []
    new_label_paths: list[Path] = []

    for product in products:
        product_code = product["product_code"]
        product_name = product["product_name"]
        has_images = product.get("has_images", False)
        image_types = product.get("image_types", [])

        if not has_images or not image_types:
            skipped_no_images += 1
            continue

        # Look up category_id by product name
        cat_id = name_to_cat_id.get(product_name)
        if cat_id is None:
            skipped_no_match += 1
            print(f"  SKIP (no category match): {product_name}")
            continue

        # Process each angle/image type
        for angle in image_types:
            src_image = PRODUCT_IMAGES_DIR / product_code / f"{angle}.jpg"
            if not src_image.exists():
                # Try .jpeg extension
                src_image = PRODUCT_IMAGES_DIR / product_code / f"{angle}.jpeg"
                if not src_image.exists():
                    print(f"  WARNING: image not found for {product_code}/{angle}")
                    continue

            # Output names: ref_{product_code}_{angle}.jpg / .txt
            out_stem = f"ref_{product_code}_{angle}"
            dst_image = YOLO_TRAIN_IMAGES / f"{out_stem}.jpg"
            dst_label = YOLO_TRAIN_LABELS / f"{out_stem}.txt"

            # Copy image
            shutil.copy2(str(src_image), str(dst_image))

            # Write YOLO label: full-image bounding box
            dst_label.write_text(f"{cat_id} 0.500000 0.500000 1.000000 1.000000\n")

            new_image_paths.append(dst_image)
            new_label_paths.append(dst_label)
            images_added += 1
            categories_covered.add(cat_id)

        if images_added % 100 == 0 and images_added > 0:
            print(f"  Progress: {images_added} images added ...")

    # Summary
    print("\n--- Summary ---")
    print(f"  Images added:          {images_added}")
    print(f"  Categories covered:    {len(categories_covered)} / {len(name_to_cat_id)}")
    print(f"  Skipped (no match):    {skipped_no_match}")
    print(f"  Skipped (no images):   {skipped_no_images}")

    if not new_image_paths:
        print("\nNo new images to upload. Done.")
        return

    # Upload to GCS
    print(f"\nUploading {len(new_image_paths)} images to gs://{GCS_BUCKET}/{GCS_IMAGES_PREFIX} ...")
    upload_to_gcs(new_image_paths, GCS_BUCKET, GCS_IMAGES_PREFIX)

    print(f"Uploading {len(new_label_paths)} labels to gs://{GCS_BUCKET}/{GCS_LABELS_PREFIX} ...")
    upload_to_gcs(new_label_paths, GCS_BUCKET, GCS_LABELS_PREFIX)

    print("\nDone! All reference images and labels uploaded to GCS.")


if __name__ == "__main__":
    main()
