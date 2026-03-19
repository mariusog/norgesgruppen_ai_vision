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
MODEL_PATH = "weights/model.pt"

# Number of detection categories (IDs 0-356; category 356 = "unknown_product")
NUM_CLASSES = 357

# YOLO input resolution -- higher = more accurate, slower
IMAGE_SIZE = 640

# ---------------------------------------------------------------------------
# Competition: inference tuning
# ---------------------------------------------------------------------------

# Minimum confidence to include a detection in output
# Lower = more detections (better recall), higher = fewer false positives
# Tune against validation mAP; start at 0.25
CONFIDENCE_THRESHOLD = 0.25

# NMS IoU threshold -- detections with IoU > this are suppressed as duplicates
IOU_THRESHOLD = 0.45

# Max detections per image -- L4 has plenty of memory, but cap for safety
MAX_DETECTIONS_PER_IMAGE = 300

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
# System limits (generic safeguards)
# ---------------------------------------------------------------------------

# Maximum iterations for search/exploration functions (prevents unbounded loops)
MAX_SEARCH_STEPS = 10_000

# Maximum file/data size to process in one pass
MAX_BATCH_SIZE = 1_000

# Cache size limit for LRU caches (0 = unbounded, use with caution)
DEFAULT_CACHE_SIZE = 1024

# Timeout budget for real-time operations (seconds)
OPERATION_TIMEOUT = 2.0
