#!/usr/bin/env bash
# Launch ALL training configs in parallel on Vertex AI.
# Unlimited credits = try everything, pick the winner.
#
# Usage: bash scripts/launch_all_training.sh

set -euo pipefail

REGION="us-central1"
PROJECT="YOUR_GCP_PROJECT_ID"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

CONFIGS=(
  "training/vertex-job-yolov8m-640.yaml"
  "training/vertex-job-yolov8m-1280.yaml"
  "training/vertex-job-yolov8l-640.yaml"
  "training/vertex-job-yolov8l-1280.yaml"
  "training/vertex-job-yolov8x-640.yaml"
  "training/vertex-job-yolov8x-1280.yaml"
)

echo "=== Launching ${#CONFIGS[@]} training jobs in parallel ==="
echo "Region: $REGION"
echo ""

for config in "${CONFIGS[@]}"; do
  name=$(basename "$config" .yaml | sed 's/vertex-job-//')
  display="${name}-${TIMESTAMP}"
  echo "Launching: $display"
  gcloud ai custom-jobs create \
    --region="$REGION" \
    --project="$PROJECT" \
    --display-name="$display" \
    --config="$config" \
    2>&1 | grep -E "submitted|ERROR" || true
  echo ""
done

echo "=== All jobs submitted. Monitor with: ==="
echo "gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --limit=10 --format='table(displayName,state,updateTime)'"
