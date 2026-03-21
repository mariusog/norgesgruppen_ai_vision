"""Named constants and tuning parameters.

All numeric thresholds, limits, and configuration values live here.
No magic numbers in logic code -- reference these constants instead.

Naming convention: UPPER_SNAKE_CASE
Sections: group related constants under comment headers.

When adding a new constant:
1. Choose a descriptive name that explains WHAT it controls
2. Add a comment explaining WHY this value was chosen
3. If the value was tuned empirically, note the benchmark that validated it
"""

# ---------------------------------------------------------------------------
# Competition: hardware constraints
# ---------------------------------------------------------------------------

# NVIDIA L4 GPU VRAM (GB) -- sandbox hardware
MAX_GPU_MEMORY_GB = 24

# System RAM (GB) -- sandbox limit
MAX_RAM_GB = 8

# Wall-clock timeout for the entire test set (seconds)
INFERENCE_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Competition: model configuration
# ---------------------------------------------------------------------------

# Path to YOLOv8 weights file (relative to repo root)
# Fallback single-model path (used when ENSEMBLE_WEIGHTS is empty)
MODEL_PATH = "weights/yolov8l-640-aug.pt"

# Number of detection categories (IDs 0-355; nc=356 in data.yaml)
# Category 355 = "unknown_product". NUM_CLASSES matches data.yaml nc.
NUM_CLASSES = 356

# YOLO input resolution -- higher = more accurate, slower
IMAGE_SIZE = 640

# Use FP16 (half precision) for inference -- ~2x faster on L4 with negligible accuracy loss
HALF_PRECISION = True

# Path to TensorRT engine file (preferred over .pt for local benchmarking).
# NOTE: .engine is NOT an allowed file type in competition submissions.
# For submission, use .pt or .onnx. This path is for local speed testing only.
MODEL_ENGINE_PATH = "weights/model.engine"

# ---------------------------------------------------------------------------
# Model ensemble — WBF (Weighted Box Fusion)
# ---------------------------------------------------------------------------

# List of weight paths to load (empty = single model mode using MODEL_PATH).
# When populated, predictions from all models are merged with WBF.
ENSEMBLE_WEIGHTS: list[str] = [
    "weights/yolov8l-1280-corrected.pt",
    "weights/yolov8l-640-aug.pt",
]

# Per-model input resolution for mixed-resolution ensembles.
# Must be same length as ENSEMBLE_WEIGHTS. If empty, all models use IMAGE_SIZE.
ENSEMBLE_IMAGE_SIZES: list[int] = [1280, 640]

# WBF IoU threshold — boxes with IoU above this are fused together
WBF_IOU_THRESHOLD = 0.55

# WBF minimum score to keep a box before fusion
WBF_SKIP_BOX_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# Two-stage classifier — refines YOLO category predictions
# When classifier weights exist, each YOLO detection is cropped, resized to
# CLASSIFIER_INPUT_SIZE, and re-classified with higher accuracy.
# ---------------------------------------------------------------------------

CLASSIFIER_PATH = "weights/classifier.pt"
CLASSIFIER_MODEL_NAME = "efficientnet_b3"

# Classifier ensemble: list of (path, model_name) tuples for multiple classifiers.
# Empty = single classifier mode using CLASSIFIER_PATH + CLASSIFIER_MODEL_NAME.
# When populated, softmax outputs are averaged across all classifiers.
CLASSIFIER_ENSEMBLE: list[tuple[str, str]] = [
    # ("weights/classifier.pt", "efficientnet_b3"),
    # ("weights/classifier2.pt", "convnext_small.fb_in22k_ft_in1k"),
]
# EfficientNet-B3 native resolution is 300px; larger crops = better fine-grained accuracy
CLASSIFIER_INPUT_SIZE = 300
USE_CLASSIFIER = True  # Set False to disable two-stage even if weights exist

# Only override YOLO's category when classifier softmax confidence exceeds this.
# Prevents low-confidence classifier predictions from overriding correct YOLO labels.
CLASSIFIER_CONFIDENCE_GATE = 0.15

# Classifier TTA: run crops through classifier with augmentations and average softmax
USE_CLASSIFIER_TTA = True

# Score fusion: blend YOLO detection confidence with classifier confidence
# final_score = SCORE_FUSION_ALPHA * yolo_score + (1 - alpha) * classifier_conf
# Set to 1.0 to disable (keep YOLO score only)
SCORE_FUSION_ALPHA = 0.7

# ---------------------------------------------------------------------------
# Prototype matching — cosine similarity against reference product embeddings
# Falls back to prototype matching when classifier confidence is low.
# ---------------------------------------------------------------------------

PROTOTYPE_PATH = "weights/prototypes.pt"
USE_PROTOTYPE_MATCHING = True
# Use prototype matching when classifier softmax is below this threshold
PROTOTYPE_CONFIDENCE_THRESHOLD = 0.5
# Minimum cosine similarity to trust a prototype match
PROTOTYPE_SIMILARITY_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Competition: inference tuning
# ---------------------------------------------------------------------------

# Minimum confidence to include a detection in output
# Lower = more detections (better recall) for mAP evaluation
# Competition mAP benefits from high recall; very low threshold lets the
# precision-recall curve be computed over the full range
CONFIDENCE_THRESHOLD = 0.01

# NMS IoU threshold -- detections with IoU > this are suppressed as duplicates
IOU_THRESHOLD = 0.45

# Test-Time Augmentation -- runs predict on flipped/scaled variants and merges
# Improves accuracy ~1-3% but ~2-3x slower. Within 300s budget at 640 with ensemble.
USE_TTA = True

# Batch size for inference -- balances GPU utilization vs memory on L4 (24 GB)
INFERENCE_BATCH_SIZE = 16

# Max detections per image -- shelf images can have 200+ products; with low
# confidence threshold we need headroom for the full precision-recall curve
MAX_DETECTIONS_PER_IMAGE = 1000

# ---------------------------------------------------------------------------
# Competition: submission constraints
# ---------------------------------------------------------------------------

# Maximum total size of all weight files in the zip (MB)
MAX_WEIGHT_SIZE_MB = 420  # Also: max 3 weight files, max 10 .py files in zip

# Safety margin: flag if projected total inference time exceeds this (seconds)
INFERENCE_BUDGET_SOFT_LIMIT = 250

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

# Supported image extensions for input directory scanning
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ---------------------------------------------------------------------------
# GCP configuration
# ---------------------------------------------------------------------------

GCP_PROJECT_ID = "ai-nm26osl-1792"
GCS_BUCKET = "ai-nm26osl-1792-nmiai"
GCS_DATASET_PREFIX = "datasets"
GCS_WEIGHTS_PREFIX = "weights"

# ---------------------------------------------------------------------------
# Classifier inference
# ---------------------------------------------------------------------------

# Batch size for classifier crop inference (balances GPU memory vs throughput)
CLASSIFIER_BATCH_SIZE = 64
