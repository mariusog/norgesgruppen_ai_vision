"""
Vertex AI training job for EfficientNet-B3 product classifier.

Second stage of two-stage detection pipeline:
  YOLO detects product bounding boxes -> this classifier identifies the product category.

Trains on:
  1. Reference product images (1 product per image, full-frame)
  2. Cropped product annotations from shelf images (YOLO bbox crops)

Usage (local test):
    python training/train_classifier.py --epochs 10 --no-upload

Usage (Vertex AI):
    See training/vertex-job-classifier.yaml
"""

from __future__ import annotations

import argparse
import functools
import time
from collections import Counter
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# PyTorch 2.6 defaults torch.load(weights_only=True) which breaks
# timm loading pretrained weights. Patch for training only.
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GCS_BUCKET = "YOUR_GCS_BUCKET"
WEIGHTS_PREFIX = "weights"
GCS_WEIGHTS_NAME = "classifier_efficientnet_b3.pt"
LOCAL_WEIGHTS_PATH = Path("weights/classifier.pt")

NUM_CLASSES = 356
INPUT_SIZE = 300  # EfficientNet-B3 native resolution; was 224 (too small for fine-grained)
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4  # Lower LR for better fine-tuning convergence
NUM_WORKERS = 4

# Dataset paths (relative to data root)
YOLO_DIR = "yolo"
IMAGES_DIR = "images"

# Data root depends on environment
LOCAL_DATA_ROOT = Path("training/data")
VERTEX_DATA_ROOT = Path("/workspace/data")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_data_root() -> Path:
    """Return the dataset root, preferring Vertex AI workspace if it exists."""
    if VERTEX_DATA_ROOT.exists():
        return VERTEX_DATA_ROOT
    return LOCAL_DATA_ROOT


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file. Returns list of (class_id, cx, cy, w, h) normalized."""
    annotations = []
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


def crop_bbox_from_image(
    img: Image.Image, cx: float, cy: float, w: float, h: float, min_size: int = 10
) -> Image.Image | None:
    """Crop a YOLO-format bbox (normalized) from a PIL image. Returns None if too small."""
    img_w, img_h = img.size
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return img.crop((x1, y1, x2, y2))


class ProductClassificationDataset(Dataset):
    """Dataset combining reference images and shelf image crops for classification."""

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path | tuple[Path, int], int, str]] = []
        # samples entries:
        #   (image_path, class_id, "ref")         — reference image, load directly
        #   ((shelf_path, ann_idx), class_id, "crop")  — shelf crop, lazy-load and crop

        self._collect_reference_images()
        self._collect_shelf_crops()

        print(f"[{split}] Total samples: {len(self.samples)} (ref + shelf crops)")

    def _collect_reference_images(self) -> None:
        """Collect reference product images from the YOLO train split.

        Only included in the training set — val split uses shelf crops only
        for unbiased evaluation.
        """
        if self.split != "train":
            print(f"  [{self.split}] Reference images: 0 (skipped for val)")
            return

        ref_dir = self.data_root / YOLO_DIR / "train" / "images"
        label_dir = self.data_root / YOLO_DIR / "train" / "labels"
        if not ref_dir.exists():
            print(f"  Warning: ref image dir not found: {ref_dir}")
            return

        count = 0
        for img_path in sorted(ref_dir.glob("ref_*")):
            label_path = label_dir / (img_path.stem + ".txt")
            annotations = parse_yolo_label(label_path)
            if not annotations:
                continue
            # Reference images have a single annotation covering the whole image
            class_id = annotations[0][0]
            if 0 <= class_id < NUM_CLASSES:
                self.samples.append((img_path, class_id, "ref"))
                count += 1

        print(f"  [{self.split}] Reference images: {count}")

    def _collect_shelf_crops(self) -> None:
        """Collect product crops from shelf images (YOLO annotations)."""
        split_dir = self.data_root / YOLO_DIR / self.split
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        if not img_dir.exists():
            print(f"  Warning: shelf image dir not found: {img_dir}")
            return

        count = 0
        for img_path in sorted(img_dir.glob("img_*")):
            label_path = label_dir / (img_path.stem + ".txt")
            annotations = parse_yolo_label(label_path)
            for ann_idx, (class_id, _, _, _, _) in enumerate(annotations):
                if 0 <= class_id < NUM_CLASSES:
                    self.samples.append(((img_path, ann_idx), class_id, "crop"))
                    count += 1

        print(f"  [{self.split}] Shelf crops: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.samples[idx]
        source, class_id, kind = entry[0], entry[1], entry[2]

        if kind == "ref":
            img = Image.open(source).convert("RGB")
        else:
            # Shelf crop: source is (img_path, ann_idx)
            img_path, ann_idx = source
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            annotations = parse_yolo_label(label_path)
            _, cx, cy, w, h = annotations[ann_idx]
            full_img = Image.open(img_path).convert("RGB")
            img = crop_bbox_from_image(full_img, cx, cy, w, h)
            if img is None:
                # Fallback: return a small black image (will be resized by transform)
                img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, class_id


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(INPUT_SIZE * 1.14)),  # 256
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# GCS helpers (reuse patterns from train.py)
# ---------------------------------------------------------------------------


def download_dataset() -> None:
    """Pull dataset from GCS if running on Vertex AI."""
    workspace = Path("/workspace")
    if not workspace.exists():
        print("Skipping GCS download — not running in Vertex AI container")
        return

    data_dir = VERTEX_DATA_ROOT
    if data_dir.exists() and any(data_dir.rglob("*.jpg")):
        print(f"Dataset already present at {data_dir}")
        return

    print(f"Pulling dataset from gs://{GCS_BUCKET}/datasets/ → {data_dir}")
    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    # Download YOLO data
    for prefix in ["datasets/yolo/", "datasets/images/"]:
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            # Map datasets/yolo/... -> data/yolo/...
            # Map datasets/images/... -> data/images/...
            rel = Path(blob.name).relative_to("datasets")
            dest = data_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
        print(f"Downloaded {len(blobs)} files from {prefix}")


def upload_weights(weights_path: Path) -> None:
    """Upload classifier weights to GCS."""
    if not weights_path.exists():
        print(f"WARNING: weights not found at {weights_path}")
        return

    size_mb = weights_path.stat().st_size / 1024 / 1024
    print(f"Uploading {weights_path} ({size_mb:.1f} MB) to GCS...")

    from google.cloud import storage  # type: ignore[import]

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    dest_path = f"{WEIGHTS_PREFIX}/{GCS_WEIGHTS_NAME}"
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(str(weights_path))
    print(f"Uploaded to gs://{GCS_BUCKET}/{dest_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.1f}%"
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def main() -> None:
    global INPUT_SIZE, GCS_WEIGHTS_NAME

    parser = argparse.ArgumentParser(description="Classifier training for NorgesGruppen products")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b3",
        help="timm model name (e.g. convnext_small.fb_in22k_ft_in1k)",
    )
    parser.add_argument(
        "--input-size", type=int, default=INPUT_SIZE, help="Input image size for classifier"
    )
    parser.add_argument(
        "--use-focal-loss", action="store_true", help="Use focal loss instead of CrossEntropy"
    )
    parser.add_argument(
        "--gcs-name", type=str, default=GCS_WEIGHTS_NAME, help="Filename for GCS upload"
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS upload")
    parser.add_argument("--run-id", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # Download dataset if on Vertex AI
    download_dataset()

    data_root = get_data_root()
    print(f"Data root: {data_root}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override globals from CLI args
    INPUT_SIZE = args.input_size
    GCS_WEIGHTS_NAME = args.gcs_name

    # Build datasets
    train_dataset = ProductClassificationDataset(
        data_root, split="train", transform=get_train_transforms()
    )
    val_dataset = ProductClassificationDataset(
        data_root, split="val", transform=get_val_transforms()
    )

    # Weighted sampling: oversample rare classes for balanced training
    class_counts = Counter(cls_id for _, cls_id, _ in train_dataset.samples)
    sample_weights = [1.0 / max(class_counts[cls_id], 1) for _, cls_id, _ in train_dataset.samples]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = timm.create_model(args.model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Model: {args.model_name}, {param_count:.1f}M params, "
        f"{NUM_CLASSES} classes, input={INPUT_SIZE}px"
    )

    # Loss, optimizer, scheduler
    if args.use_focal_loss:
        # Focal loss: down-weights easy examples, focuses on hard ones
        # gamma=2.0 is the standard value from the focal loss paper
        class FocalLoss(nn.Module):
            def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
                super().__init__()
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

            def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                ce_loss = self.ce(inputs, targets)
                p_t = torch.exp(-ce_loss)
                focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
                return focal_loss.mean()

        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
        print("Using Focal Loss (gamma=2.0)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / "classifier.pt"

    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs} | LR: {lr:.6f}")
        print(f"{'=' * 60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")

        # Save best model (state_dict only for sandbox compatibility)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            size_mb = save_path.stat().st_size / 1024 / 1024
            print(f"  -> New best! Saved to {save_path} ({size_mb:.1f} MB)")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed / 60:.1f} min")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Weights saved to: {save_path}")
    print(f"{'=' * 60}")

    # Upload to GCS
    if not args.no_upload:
        upload_weights(save_path)

    print("Done.")


if __name__ == "__main__":
    main()
