"""
NM i AI 2026 - NorgesGruppen Object Detection
Competition entry point.

Usage:
    python run.py --input /path/to/images/ --output /path/to/predictions.json

Output: JSON array of {image_id, category_id, bbox, score} dicts.
        bbox format: [x, y, width, height] in pixels.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import torch

# PyTorch 2.6 defaults torch.load(weights_only=True) which breaks
# ultralytics 8.1.0 model loading. Patch before importing ultralytics.
torch.load = functools.partial(torch.load, weights_only=False)

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.constants import (  # noqa: E402
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_CONFIDENCE_GATE,
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_MODEL_NAME,
    CLASSIFIER_PATH,
    CONFIDENCE_THRESHOLD,
    ENSEMBLE_IMAGE_SIZES,
    ENSEMBLE_WEIGHTS,
    HALF_PRECISION,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    INFERENCE_BATCH_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS_PER_IMAGE,
    MODEL_ENGINE_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    PROTOTYPE_CONFIDENCE_THRESHOLD,
    PROTOTYPE_PATH,
    PROTOTYPE_SIMILARITY_THRESHOLD,
    USE_CLASSIFIER,
    USE_CLASSIFIER_TTA,
    USE_PROTOTYPE_MATCHING,
    USE_TTA,
    WBF_IOU_THRESHOLD,
    WBF_SKIP_BOX_THRESHOLD,
)
from src.prototype_matcher import load_prototypes, match_prototypes  # noqa: E402


def load_model() -> YOLO:
    """Load YOLOv8 model, preferring TensorRT engine over PyTorch weights."""
    engine = Path(MODEL_ENGINE_PATH)
    weights = Path(MODEL_PATH)

    if engine.exists():
        model_path = engine
    elif weights.exists():
        model_path = weights
    else:
        raise FileNotFoundError(f"No model found at {engine} or {weights}")

    model = YOLO(str(model_path))
    model.to("cuda")
    return model


def load_ensemble_models() -> list[YOLO]:
    """Load multiple YOLO models for ensemble inference."""
    models: list[YOLO] = []
    for weight_path in ENSEMBLE_WEIGHTS:
        p = Path(weight_path)
        if not p.exists():
            raise FileNotFoundError(f"Ensemble weight not found: {p}")
        model = YOLO(str(p))
        model.to("cuda")
        models.append(model)
    return models


def load_classifier() -> torch.nn.Module | None:
    """Load the two-stage classifier if weights exist and USE_CLASSIFIER is True.

    Returns the classifier model on CUDA in eval mode, or None if unavailable.
    """
    if not USE_CLASSIFIER:
        return None

    classifier_path = Path(CLASSIFIER_PATH)
    if not classifier_path.exists():
        return None

    import timm

    model = timm.create_model(CLASSIFIER_MODEL_NAME, num_classes=NUM_CLASSES)
    state_dict = torch.load(str(classifier_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model


def classify_crops(
    image_paths: list[Path],
    predictions: list[dict],
    classifier: torch.nn.Module,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
) -> list[dict]:
    """Re-classify each YOLO detection using the two-stage classifier.

    Crops each detected bbox from the original image, resizes to
    CLASSIFIER_INPUT_SIZE, runs the classifier, and overrides category_id.

    Args:
        image_paths: list of image paths used during detection.
        predictions: list of prediction dicts (mutated in place and returned).
        classifier: the loaded classifier model.
        batch_size: number of crops to classify per forward pass.

    Returns:
        The updated predictions list with refined category_id values.
    """
    if not predictions:
        return predictions

    # Build a lookup from image_id to file path
    id_to_path: dict[int, Path] = {}
    for p in image_paths:
        img_id = int(p.stem.split("_")[-1])
        id_to_path[img_id] = p

    # ImageNet normalization transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Cache opened images to avoid re-reading the same file
    image_cache: dict[int, Image.Image] = {}

    # Prepare all crops
    crop_tensors: list[torch.Tensor] = []
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in image_cache:
            img_path = id_to_path[img_id]
            image_cache[img_id] = Image.open(str(img_path)).convert("RGB")
        img = image_cache[img_id]

        # bbox is [x, y, width, height] in pixels
        bx, by, bw, bh = pred["bbox"]
        # Add 10% padding for better classifier context
        pad_w = bw * 0.10
        pad_h = bh * 0.10
        img_w, img_h = img.size
        left = max(0, int(bx - pad_w))
        upper = max(0, int(by - pad_h))
        right = min(img_w, int(bx + bw + pad_w))
        lower = min(img_h, int(by + bh + pad_h))

        crop = img.crop((left, upper, right, lower))
        crop = crop.resize(
            (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE),
            Image.BILINEAR,
        )

        # Convert to tensor [C, H, W] in [0, 1] range, then normalize
        tensor = transforms.functional.to_tensor(crop)
        tensor = normalize(tensor)
        crop_tensors.append(tensor)

    # Free cached images
    image_cache.clear()

    # Load prototypes if available
    prototypes = None
    proto_path = Path(PROTOTYPE_PATH)
    if USE_PROTOTYPE_MATCHING and proto_path.exists():
        prototypes = load_prototypes(proto_path)
        prototypes["embeddings"] = prototypes["embeddings"].to("cuda")

    # Run classifier in batches with optional TTA + prototype features
    all_class_ids: list[int] = []
    all_confidences: list[float] = []
    all_features: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(crop_tensors), batch_size):
            batch = torch.stack(crop_tensors[i : i + batch_size]).to("cuda")

            if USE_CLASSIFIER_TTA:
                # TTA: original + horizontal flip, average softmax
                logits_orig = classifier(batch)
                logits_flip = classifier(torch.flip(batch, dims=[3]))
                probs = (
                    torch.nn.functional.softmax(logits_orig, dim=1)
                    + torch.nn.functional.softmax(logits_flip, dim=1)
                ) / 2.0
            else:
                logits = classifier(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)

            max_probs, class_ids = probs.max(dim=1)
            all_class_ids.extend(class_ids.cpu().tolist())
            all_confidences.extend(max_probs.cpu().tolist())

            # Extract features for prototype matching if needed
            if prototypes is not None:
                feats = classifier.forward_features(batch)
                if feats.dim() == 4:
                    feats = feats.mean(dim=[2, 3])
                feats = torch.nn.functional.normalize(feats, dim=1)
                all_features.append(feats.cpu())

    # Override YOLO category using classifier + prototype fallback
    for idx, (pred, cls_id, conf) in enumerate(
        zip(predictions, all_class_ids, all_confidences, strict=True)
    ):
        if conf >= CLASSIFIER_CONFIDENCE_GATE:
            pred["category_id"] = int(cls_id)

            # For mid-confidence predictions, try prototype matching
            if prototypes is not None and conf < PROTOTYPE_CONFIDENCE_THRESHOLD:
                feat_idx = idx
                batch_idx = feat_idx // batch_size
                within_batch = feat_idx % batch_size
                if batch_idx < len(all_features):
                    feat = all_features[batch_idx][within_batch : within_batch + 1].to("cuda")
                    sims, proto_ids = match_prototypes(feat, prototypes)
                    if sims[0, 0].item() >= PROTOTYPE_SIMILARITY_THRESHOLD:
                        pred["category_id"] = int(proto_ids[0, 0].item())

    return predictions


def collect_images(input_dir: Path) -> list[Path]:
    """Return sorted list of image paths from the input directory."""
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_inference(model: YOLO, image_paths: list[Path]) -> list[dict]:
    """Run detection on all images in batches, returning competition-format predictions."""
    predictions: list[dict] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), INFERENCE_BATCH_SIZE):
            batch_paths = image_paths[i : i + INFERENCE_BATCH_SIZE]
            batch_strs = [str(p) for p in batch_paths]
            batch_ids = [int(p.stem.split("_")[-1]) for p in batch_paths]

            results = model.predict(
                batch_strs,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                half=HALF_PRECISION,
                augment=USE_TTA,
                max_det=MAX_DETECTIONS_PER_IMAGE,
            )

            for image_id, result in zip(batch_ids, results, strict=True):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(box.cls[0].item()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(box.conf[0].item()),
                        }
                    )

    return predictions


def run_ensemble_inference(models: list[YOLO], image_paths: list[Path]) -> list[dict]:
    """Run ensemble detection using WBF to merge predictions from multiple models.

    Each image is processed independently: all models predict on it, then
    Weighted Box Fusion merges the detections into a single set.
    """
    from ensemble_boxes import weighted_boxes_fusion

    predictions: list[dict] = []
    with torch.no_grad():
        for img_path in image_paths:
            image_id = int(img_path.stem.split("_")[-1])
            img_str = str(img_path)

            boxes_list: list[list[list[float]]] = []
            scores_list: list[list[float]] = []
            labels_list: list[list[int]] = []
            img_h: int = 0
            img_w: int = 0

            for model_idx, model in enumerate(models):
                # Use per-model resolution if configured, else fall back to IMAGE_SIZE
                model_imgsz = (
                    ENSEMBLE_IMAGE_SIZES[model_idx]
                    if ENSEMBLE_IMAGE_SIZES and model_idx < len(ENSEMBLE_IMAGE_SIZES)
                    else IMAGE_SIZE
                )
                results = model.predict(
                    img_str,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=model_imgsz,
                    half=HALF_PRECISION,
                    augment=USE_TTA,
                    max_det=MAX_DETECTIONS_PER_IMAGE,
                )
                result = results[0]
                img_h, img_w = result.orig_shape

                model_boxes: list[list[float]] = []
                model_scores: list[float] = []
                model_labels: list[int] = []

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Normalize to [0, 1] for WBF
                        model_boxes.append(
                            [
                                x1 / img_w,
                                y1 / img_h,
                                x2 / img_w,
                                y2 / img_h,
                            ]
                        )
                        model_scores.append(float(box.conf[0].item()))
                        model_labels.append(int(box.cls[0].item()))

                boxes_list.append(model_boxes)
                scores_list.append(model_scores)
                labels_list.append(model_labels)

            # Skip WBF if no model produced any detections
            if all(len(b) == 0 for b in boxes_list):
                continue

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=None,
                iou_thr=WBF_IOU_THRESHOLD,
                skip_box_thr=WBF_SKIP_BOX_THRESHOLD,
            )

            for box, score, label in zip(fused_boxes, fused_scores, fused_labels, strict=True):
                # Denormalize back to pixel coordinates (xyxy)
                x1 = float(box[0]) * img_w
                y1 = float(box[1]) * img_h
                x2 = float(box[2]) * img_w
                y2 = float(box[3]) * img_h
                # Convert xyxy → xywh for competition format
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="NorgesGruppen grocery detector")
    parser.add_argument("--input", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    t_start = time.perf_counter()

    image_paths = collect_images(input_dir)
    classifier = load_classifier()

    if ENSEMBLE_WEIGHTS:
        models = load_ensemble_models()
        predictions = run_ensemble_inference(models, image_paths)
    else:
        model = load_model()
        predictions = run_inference(model, image_paths)

    if classifier is not None:
        predictions = classify_crops(image_paths, predictions, classifier)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions))

    elapsed = time.perf_counter() - t_start
    print(f"Processed {len(image_paths)} images in {elapsed:.1f}s → {len(predictions)} detections")


if __name__ == "__main__":
    main()
