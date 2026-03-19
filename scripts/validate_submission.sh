#!/usr/bin/env bash
# Pre-submission validation checklist.
# Run before creating the competition zip.
#
# Usage: bash scripts/validate_submission.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

check() {
  local desc="$1"
  local cmd="$2"
  if eval "$cmd" > /dev/null 2>&1; then
    echo "  OK  $desc"
    ((PASS++)) || true
  else
    echo " FAIL $desc"
    ((FAIL++)) || true
  fi
}

echo "=== NM i AI 2026 Submission Checklist ==="
echo ""

# 1. Forbidden imports
echo "[1] Security scan (forbidden imports)"
VIOLATIONS=$(grep -rPn '^\s*(import (os|subprocess|socket)\b|from (os|subprocess|socket)(\s|\.|;|$))' \
  "$REPO_ROOT/run.py" "$REPO_ROOT/src/" 2>/dev/null || true)
if [[ -z "$VIOLATIONS" ]]; then
  echo "  OK  No forbidden imports (os, subprocess, socket)"
  ((PASS++)) || true
else
  echo " FAIL Forbidden imports found:"
  echo "$VIOLATIONS" | sed 's/^/       /'
  ((FAIL++)) || true
fi

echo ""

# 2. Entry point
echo "[2] Entry point structure"
check "run.py exists at root" "test -f '$REPO_ROOT/run.py'"
check "run.py has --input arg" "grep -q '\-\-input' '$REPO_ROOT/run.py'"
check "run.py has --output arg" "grep -q '\-\-output' '$REPO_ROOT/run.py'"
check "run.py uses torch.no_grad()" "grep -q 'torch.no_grad' '$REPO_ROOT/run.py'"
check "run.py uses pathlib" "grep -q 'pathlib' '$REPO_ROOT/run.py'"

echo ""

# 3. Weight file size
echo "[3] Weight file size (limit: 420 MB)"
WEIGHTS_DIR="$REPO_ROOT/weights"
if [[ -d "$WEIGHTS_DIR" ]]; then
  TOTAL_BYTES=$(find "$WEIGHTS_DIR" -type f \( -name "*.pt" -o -name "*.onnx" -o -name "*.engine" \) \
    -exec stat -c%s {} \; | awk '{sum+=$1} END{print sum+0}')
  TOTAL_MB=$((TOTAL_BYTES / 1024 / 1024))
  if [[ $TOTAL_MB -le 420 ]]; then
    echo "  OK  Weight files: ${TOTAL_MB} MB / 420 MB"
    ((PASS++)) || true
  else
    echo " FAIL Weight files: ${TOTAL_MB} MB EXCEEDS 420 MB limit"
    ((FAIL++)) || true
  fi
else
  echo " WARN weights/ directory not found — run model-agent to train and download weights"
fi

echo ""

# 4. Tests
echo "[4] Test suite"
check "Security tests pass" "python -m pytest '$REPO_ROOT/tests/test_security.py' -q --tb=line 2>&1 | tail -5 | grep -v FAILED"
check "Format tests pass" "python -m pytest '$REPO_ROOT/tests/test_output_format.py' -q --tb=line 2>&1 | tail -5 | grep -v FAILED"

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [[ $FAIL -gt 0 ]]; then
  echo "Fix the failures above before submitting."
  exit 1
fi
echo "All checks passed. Ready to zip for submission."
exit 0
