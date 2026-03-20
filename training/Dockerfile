# Training container for NM i AI 2026 — NorgesGruppen grocery detection
# Pushed to: europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest
#
# Build and push:
#   gcloud builds submit --config=cloudbuild.yaml --project=ai-nm26osl-1792
#
# Run locally (requires GPU):
#   docker run --gpus all -v $(pwd):/workspace \
#     europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest \
#     python training/train.py --data training/data.yaml --epochs 10 --no-upload

# Google Deep Learning Container — PyTorch 2.6 + Python 3.11 + CUDA 12.4 pre-installed
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-6.py311

WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY training/ training/

# Only install what's NOT already in the DLC base image.
# DLC includes: torch, torchvision, numpy, scipy, Pillow, etc.
# We need: ultralytics, google-cloud-storage, and a few competition-specific packages.
RUN pip install --no-cache-dir \
    "ultralytics==8.1.0" \
    "google-cloud-storage==2.14.0" \
    "pycocotools==2.0.7" \
    "albumentations==1.3.1" \
    "ensemble-boxes==1.0.9"

# Default command: run training with standard config
CMD ["python", "training/train.py", "--data", "training/data.yaml"]
