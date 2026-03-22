#!/usr/bin/env bash
# Launch training jobs on Vertex AI using config files.
#
# Usage:
#   bash scripts/launch_training.sh baseline     # YOLOv8m, 640px (default)
#   bash scripts/launch_training.sh large         # YOLOv8l, 1280px (top-dawg)

set -euo pipefail

REGION="us-central1"
PROJECT="ai-nm26osl-1792"
PROFILE="${1:-baseline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$PROFILE" in
  baseline)
    # Default CMD in container handles this — no config file needed
    DISPLAY_NAME="yolov8m-640-$(date +%Y%m%d-%H%M%S)"
    echo "=== Launching: $DISPLAY_NAME (YOLOv8m, 640px) ==="
    gcloud ai custom-jobs create \
      --region="$REGION" \
      --project="$PROJECT" \
      --display-name="$DISPLAY_NAME" \
      --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest
    ;;
  large)
    DISPLAY_NAME="yolov8l-1280-$(date +%Y%m%d-%H%M%S)"
    echo "=== Launching: $DISPLAY_NAME (YOLOv8l, 1280px) ==="
    gcloud ai custom-jobs create \
      --region="$REGION" \
      --project="$PROJECT" \
      --display-name="$DISPLAY_NAME" \
      --config="$SCRIPT_DIR/training/vertex-job-yolov8l-1280.yaml"
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    echo "Usage: $0 {baseline|large}"
    exit 1
    ;;
esac

echo ""
echo "=== Monitor with: ==="
echo "gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --limit=3 --format='table(displayName,state)'"
