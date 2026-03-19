#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Norgesgruppen AI Vision devcontainer ==="

# Install Python dev tools + competition packages
pip install --upgrade pip

# Dev tooling
pip install pytest pytest-cov ruff mypy bandit pip-audit

# Competition packages — pinned to exact sandbox versions to catch version-mismatch bugs early.
# CPU variants used locally (sandbox uses CUDA builds, but all Python APIs are identical).
pip install \
  ultralytics==8.1.0 \
  torch==2.6.0 \
  torchvision==0.21.0 \
  onnxruntime==1.20.0 \
  opencv-python-headless==4.9.0.80 \
  albumentations==1.3.1 \
  Pillow==10.2.0 \
  "numpy<2.0" \
  scipy==1.12.0 \
  scikit-learn==1.4.0 \
  pycocotools==2.0.7 \
  ensemble-boxes==1.0.9 \
  timm==0.9.12 \
  supervision==0.18.0 \
  safetensors==0.4.2

# GCP client library (for training/train.py dataset download)
pip install google-cloud-storage==2.14.0

# Install project in editable mode
pip install -e "." 2>/dev/null || true

# Authenticate gcloud using Application Default Credentials
# (VS Code will prompt on first use, or run: gcloud auth application-default login)
gcloud config set project ai-nm26osl-1792 2>/dev/null || true
gcloud config set compute/region europe-west4 2>/dev/null || true

echo ""
echo "=== DevContainer ready ==="
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  Ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'not found')"
echo "  gcloud project: $(gcloud config get-value project 2>/dev/null || echo 'not configured')"
echo ""
echo "Next steps:"
echo "  1. Run: gcloud auth application-default login"
echo "  2. Download training data: bash scripts/download_dataset.sh"
echo "  3. Run tests: python -m pytest tests/ -q --tb=line 2>&1 | tail -20"
