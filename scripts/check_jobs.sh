#!/bin/bash
echo "=== Classifier Job ==="
gcloud ai custom-jobs describe 5169750412189761536 --region=us-central1 --project=ai-nm26osl-1792 --format="value(state)" 2>&1
echo "=== YOLOv8x-1280 ==="
gcloud ai custom-jobs list --region=us-central1 --project=ai-nm26osl-1792 --limit=10 --format="table(displayName,state)" 2>&1 | grep -E "yolov8|classifier|STATE"
echo "=== GCS Weights ==="
gcloud storage ls gs://ai-nm26osl-1792-nmiai/weights/ 2>&1
