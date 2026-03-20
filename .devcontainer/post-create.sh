#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Norgesgruppen AI Vision devcontainer ==="

# Install gcloud CLI via official apt repo (Debian bookworm)
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update -qq && sudo apt-get install -y -q google-cloud-cli

# Install project + all dependencies in one pass (pyproject.toml is the single source of truth)
pip install --upgrade pip
pip install -e ".[dev]"

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
echo "  node version: $(node --version 2>/dev/null || echo 'not found')"
echo ""
echo "Next steps:"
echo "  1. Run: gcloud auth application-default login"
echo "  2. Download training data: bash scripts/download_dataset.sh"
echo "  3. Run tests: python -m pytest tests/ -q --tb=line 2>&1 | tail -20"
