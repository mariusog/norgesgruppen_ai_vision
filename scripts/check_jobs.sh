#!/bin/bash
echo "=== Classifier Job ==="
gcloud ai custom-jobs describe YOUR_JOB_ID --region=us-central1 --project=YOUR_GCP_PROJECT_ID --format="value(state)" 2>&1
echo "=== YOLOv8x-1280 ==="
gcloud ai custom-jobs list --region=us-central1 --project=YOUR_GCP_PROJECT_ID --limit=10 --format="table(displayName,state)" 2>&1 | grep -E "yolov8|classifier|STATE"
echo "=== GCS Weights ==="
gcloud storage ls gs://YOUR_GCS_BUCKET/weights/ 2>&1
