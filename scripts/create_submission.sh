#!/usr/bin/env bash
# Create competition submission zip.
# Includes run.py, src/, and weights/ — nothing else.
#
# Usage: bash scripts/create_submission.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$REPO_ROOT/submission.zip"

echo "=== Creating Submission Zip ==="
echo ""

# Pre-flight checks
echo "[1] Pre-flight checks"
if [[ ! -f "$REPO_ROOT/run.py" ]]; then
  echo " FAIL run.py not found at repo root"
  exit 1
fi
echo "  OK  run.py exists"

if [[ ! -d "$REPO_ROOT/weights" ]] || [[ -z "$(find "$REPO_ROOT/weights" -name '*.pt' -o -name '*.onnx' 2>/dev/null)" ]]; then
  echo " FAIL No weight files found in weights/"
  echo "      Run: bash scripts/download_weights.sh"
  exit 1
fi
echo "  OK  Weight files found"
echo ""

# Run validation
echo "[2] Running validation"
bash "$REPO_ROOT/scripts/validate_submission.sh"
echo ""

# Create zip
echo "[3] Creating zip"
cd "$REPO_ROOT"
rm -f submission.zip

zip -r submission.zip \
  run.py \
  src/__init__.py \
  src/constants.py \
  weights/*.pt weights/*.onnx weights/*.safetensors 2>/dev/null \
  -x "*__pycache__*" "*egg-info*" "*.pyc"

echo ""

# Verify zip structure
echo "[4] Verifying zip contents"
unzip -l submission.zip
echo ""

ZIP_SIZE=$(stat -c%s "$OUTPUT")
ZIP_MB=$((ZIP_SIZE / 1024 / 1024))
echo "=== Submission ready: $OUTPUT (${ZIP_MB} MB) ==="
echo "Upload this file to the competition portal."
