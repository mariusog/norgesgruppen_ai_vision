#!/usr/bin/env bash
# Download trained model weights from GCS after training completes.
# Fetches the latest/best .pt file and saves it to weights/model.pt.
#
# Prerequisites: gcloud CLI authenticated, project set to ai-nm26osl-1792
#
# Usage: bash scripts/download_weights.sh

set -euo pipefail

GCS_BUCKET="ai-nm26osl-1792-nmiai"
GCS_WEIGHTS="gs://${GCS_BUCKET}/weights/"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_ROOT/weights"
DEST_FILE="$DEST_DIR/model.pt"
MAX_MB=420

echo "=== Download Weights from GCS ==="
echo ""

# 1. List available weight files
echo "[1] Listing weight files in ${GCS_WEIGHTS}"
FILE_LIST=$(gcloud storage ls -l "$GCS_WEIGHTS" 2>/dev/null || true)
if [[ -z "$FILE_LIST" ]]; then
  echo " FAIL No files found in ${GCS_WEIGHTS}"
  echo "      Has training completed and uploaded weights?"
  exit 1
fi
echo "$FILE_LIST" | head -20
echo ""

# 2. Find the latest .pt file (sorted by modification time, newest first)
echo "[2] Selecting latest .pt file"
LATEST_PT=$(gcloud storage ls "$GCS_WEIGHTS**.pt" 2>/dev/null | tail -1 || true)
if [[ -z "$LATEST_PT" ]]; then
  echo " FAIL No .pt files found in ${GCS_WEIGHTS}"
  exit 1
fi
echo "  Selected: $LATEST_PT"
echo ""

# 3. Download to weights/model.pt
echo "[3] Downloading to ${DEST_FILE}"
mkdir -p "$DEST_DIR"
gcloud storage cp "$LATEST_PT" "$DEST_FILE"
echo ""

# 4. Verify file size
echo "[4] Checking file size (limit: ${MAX_MB} MB)"
FILE_BYTES=$(stat -c%s "$DEST_FILE")
FILE_MB=$((FILE_BYTES / 1024 / 1024))
if [[ $FILE_MB -le $MAX_MB ]]; then
  echo "  OK  ${DEST_FILE}: ${FILE_MB} MB / ${MAX_MB} MB"
else
  echo " FAIL ${DEST_FILE}: ${FILE_MB} MB EXCEEDS ${MAX_MB} MB limit"
  exit 1
fi
echo ""

# 5. Summary
echo "=== Summary ==="
echo "  Source:      $LATEST_PT"
echo "  Destination: $DEST_FILE"
echo "  Size:        ${FILE_MB} MB"
echo "  Status:      Ready for inference"
