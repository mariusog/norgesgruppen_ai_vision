#!/usr/bin/env bash
# Launch a full-pipeline evaluation job on Vertex AI.
#
# Usage:
#   ./scripts/launch_eval.sh "eval-baseline" ""
#   ./scripts/launch_eval.sh "eval-no-tta" "--no-tta"
#   ./scripts/launch_eval.sh "eval-corrected-yolo" "--ensemble-weights weights/yolov8l-1280-corrected.pt,weights/yolov8l-640-aug.pt,weights/yolov8m-640-aug.pt"
#   ./scripts/launch_eval.sh "eval-with-classifier" "--classifier weights/classifier.pt"
#   ./scripts/launch_eval.sh "eval-single-model" "--no-ensemble --no-classifier"
#
# The job runs on an A100 in us-central1 using the existing trainer container.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job-name> [extra-args]"
    echo ""
    echo "Examples:"
    echo "  $0 eval-baseline"
    echo "  $0 eval-no-tta \"--no-tta\""
    echo "  $0 eval-custom \"--ensemble-weights weights/a.pt,weights/b.pt --conf 0.02\""
    exit 1
fi

JOB_NAME="${1}"
EXTRA_ARGS="${2:-}"
REGION="us-central1"
PROJECT="ai-nm26osl-1792"
CONTAINER="europe-west4-docker.pkg.dev/ai-nm26osl-1792/nmiai/trainer:latest"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DISPLAY_NAME="${JOB_NAME}-${TIMESTAMP}"

# Build the command args for the container.
# The entrypoint script is eval_full_pipeline.py with optional extra args.
CMD_ARGS="python,scripts/eval_full_pipeline.py"

# Split EXTRA_ARGS on spaces into comma-separated args for --args flag
if [ -n "$EXTRA_ARGS" ]; then
    # Convert space-separated args to comma-separated for gcloud
    COMMA_ARGS=$(echo "$EXTRA_ARGS" | tr ' ' ',')
    CMD_ARGS="${CMD_ARGS},${COMMA_ARGS}"
fi

echo "============================================================"
echo "  Launching eval job: ${DISPLAY_NAME}"
echo "  Container: ${CONTAINER}"
echo "  Region:    ${REGION}"
echo "  Command:   ${CMD_ARGS}"
echo "============================================================"
echo ""

gcloud ai custom-jobs create \
    --region="$REGION" \
    --project="$PROJECT" \
    --display-name="$DISPLAY_NAME" \
    --worker-pool-spec=machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri="$CONTAINER" \
    --args="$CMD_ARGS"

echo ""
echo "=== Job submitted: ${DISPLAY_NAME} ==="
echo ""
echo "Monitor with:"
echo "  gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --limit=5 --format='table(displayName,state)'"
echo ""
echo "Stream logs with:"
echo "  gcloud ai custom-jobs stream-logs \$(gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --limit=1 --format='value(name)') --region=$REGION --project=$PROJECT"
