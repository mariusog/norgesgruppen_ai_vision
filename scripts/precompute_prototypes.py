"""Pre-compute prototype embeddings from reference product images.

Loads a trained timm classifier, extracts penultimate-layer features from
all reference images (training/data/yolo/train/images/ref_*), averages
per category, and saves as weights/prototypes.pt.

This is a TRAINING script (not part of submission), so it may use os, etc.

Usage:
    python scripts/precompute_prototypes.py
    python scripts/precompute_prototypes.py --model-name convnext_small \
        --classifier-path weights/classifier2.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 356
DEFAULT_MODEL_NAME = "efficientnet_b3"
DEFAULT_CLASSIFIER_PATH = "weights/classifier.pt"
DEFAULT_OUTPUT_PATH = "weights/prototypes.pt"
DEFAULT_INPUT_SIZE = 300

# Data paths
LOCAL_DATA_ROOT = Path("training/data")
VERTEX_DATA_ROOT = Path("/workspace/data")
YOLO_DIR = "yolo"


def get_data_root() -> Path:
    """Return the dataset root, preferring Vertex AI workspace if it exists."""
    if VERTEX_DATA_ROOT.exists():
        return VERTEX_DATA_ROOT
    return LOCAL_DATA_ROOT


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file. Returns list of (class_id, cx, cy, w, h)."""
    annotations: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return annotations
    text = label_path.read_text().strip()
    if not text:
        return annotations
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((class_id, cx, cy, w, h))
    return annotations


def collect_reference_samples(data_root: Path) -> dict[int, list[Path]]:
    """Collect reference image paths grouped by class ID.

    Reads YOLO label files for ref_* images to determine the class ID
    for each reference image.

    Returns:
        Dict mapping class_id -> list of image paths.
    """
    ref_dir = data_root / YOLO_DIR / "train" / "images"
    label_dir = data_root / YOLO_DIR / "train" / "labels"

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference image directory not found: {ref_dir}")

    samples: dict[int, list[Path]] = defaultdict(list)
    for img_path in sorted(ref_dir.glob("ref_*")):
        label_path = label_dir / (img_path.stem + ".txt")
        annotations = parse_yolo_label(label_path)
        if not annotations:
            continue
        # Reference images have a single annotation covering the whole image
        class_id = annotations[0][0]
        if 0 <= class_id < NUM_CLASSES:
            samples[class_id].append(img_path)

    return samples


def build_transform(input_size: int) -> transforms.Compose:
    """Build inference transform for feature extraction."""
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    image_paths: list[Path],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract L2-normalized penultimate-layer features for a list of images.

    Uses model.forward_features() + global average pooling to get embeddings.

    Returns:
        Tensor of shape [len(image_paths), feature_dim], L2-normalized.
    """
    all_features: list[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors: list[torch.Tensor] = []
        for p in batch_paths:
            img = Image.open(str(p)).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)

        # Extract penultimate features using timm's forward_features
        feats = model.forward_features(batch)  # [B, C, H, W] or [B, N, C]

        # Handle different output shapes:
        if feats.dim() == 4:
            # CNN-style: [B, C, H, W] -> global average pool -> [B, C]
            feats = feats.mean(dim=[2, 3])
        elif feats.dim() == 3:
            # Transformer-style: [B, N, C] -> average over tokens -> [B, C]
            feats = feats.mean(dim=1)
        # else: [B, C] already pooled

        # L2-normalize
        feats = F.normalize(feats, dim=1)
        all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute prototype embeddings")
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=DEFAULT_CLASSIFIER_PATH,
        help="Path to trained classifier weights",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="timm model name (must match classifier training)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Input image resolution",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for prototypes.pt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load classifier
    classifier_path = Path(args.classifier_path)
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    print(f"Loading model: {args.model_name} from {classifier_path}")
    model = timm.create_model(args.model_name, num_classes=NUM_CLASSES)
    state_dict = torch.load(str(classifier_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Collect reference images
    data_root = get_data_root()
    print(f"Data root: {data_root}")
    samples_by_class = collect_reference_samples(data_root)
    total_images = sum(len(paths) for paths in samples_by_class.values())
    print(f"Found {total_images} reference images across {len(samples_by_class)} classes")

    # Build transform
    transform = build_transform(args.input_size)

    # Extract features and compute per-class prototypes
    class_ids: list[int] = []
    prototype_list: list[torch.Tensor] = []

    for class_id in sorted(samples_by_class.keys()):
        image_paths = samples_by_class[class_id]
        features = extract_features(model, image_paths, transform, device, args.batch_size)

        # Average all embeddings for this class to get the prototype
        prototype = features.mean(dim=0)
        # Re-normalize after averaging
        prototype = F.normalize(prototype, dim=0)

        class_ids.append(class_id)
        prototype_list.append(prototype)

    # Stack into a single tensor
    embeddings = torch.stack(prototype_list)  # [num_classes, feature_dim]
    feature_dim = embeddings.shape[1]

    print(f"\nPrototype matrix: {embeddings.shape} ({len(class_ids)} classes, {feature_dim}D)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prototype_data = {
        "embeddings": embeddings,
        "class_ids": class_ids,
        "model_name": args.model_name,
    }
    torch.save(prototype_data, str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved prototypes to {output_path} ({size_kb:.1f} KB)")
    print(f"Classes covered: {len(class_ids)} / {NUM_CLASSES}")

    # Sanity check: verify size is under 5MB
    size_mb = size_kb / 1024
    if size_mb > 5.0:
        print(f"WARNING: Prototype file is {size_mb:.1f} MB, exceeds 5MB target!")
    else:
        print(f"Size OK: {size_mb:.2f} MB (target: <5MB)")


if __name__ == "__main__":
    # Patch torch.load for timm compatibility
    import functools

    torch.load = functools.partial(torch.load, weights_only=False)

    main()
