#!/usr/bin/env bash
# Download competition dataset from GCS to local workspace.
# Run this once before training locally or before launching Vertex AI jobs.
#
# Prerequisites: gcloud CLI authenticated, project set to ai-nm26osl-1792
#
# Usage: bash scripts/download_dataset.sh [--dest /path/to/data]

set -euo pipefail

GCS_BUCKET="ai-nm26osl-1792-nmiai"
DEST_DIR="${1:-/tmp/nmiai_data}"

echo "Downloading dataset from gs://${GCS_BUCKET}/datasets/ → ${DEST_DIR}"
mkdir -p "$DEST_DIR"

gcloud storage cp -r "gs://${GCS_BUCKET}/datasets/*" "$DEST_DIR/"

echo "Done. Dataset at: $DEST_DIR"
echo "Update training/data.yaml path: to $DEST_DIR"
