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
BLOCKED="os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|http|multiprocessing|threading|signal|gc|code|codeop|pty"
VIOLATIONS=$(grep -rPn "^\s*(import ($BLOCKED)\b|from ($BLOCKED)(\s|\.|;|$))" \
  "$REPO_ROOT/run.py" "$REPO_ROOT/src/" 2>/dev/null || true)
if [[ -z "$VIOLATIONS" ]]; then
  echo "  OK  No forbidden imports (extended blocklist)"
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

# 5. Zip structure
echo "[5] Zip structure validation"
# Count .py files (max 10)
PY_COUNT=$(find "$REPO_ROOT/run.py" "$REPO_ROOT/src/" -name "*.py" -type f 2>/dev/null | wc -l)
if [[ $PY_COUNT -le 10 ]]; then
  echo "  OK  Python files: ${PY_COUNT} / 10 max"
  ((PASS++)) || true
else
  echo " FAIL Python files: ${PY_COUNT} EXCEEDS 10 file limit"
  ((FAIL++)) || true
fi

# Count weight files (max 3)
WEIGHT_COUNT=0
if [[ -d "$REPO_ROOT/weights" ]]; then
  WEIGHT_COUNT=$(find "$REPO_ROOT/weights" -type f \( -name "*.pt" -o -name "*.onnx" -o -name "*.engine" \) 2>/dev/null | wc -l)
fi
if [[ $WEIGHT_COUNT -le 3 ]]; then
  echo "  OK  Weight files: ${WEIGHT_COUNT} / 3 max"
  ((PASS++)) || true
else
  echo " FAIL Weight files: ${WEIGHT_COUNT} EXCEEDS 3 file limit"
  ((FAIL++)) || true
fi

# Verify run.py is at root (will be at zip root)
if [[ -f "$REPO_ROOT/run.py" ]]; then
  echo "  OK  run.py is at repository root (will be at zip root)"
  ((PASS++)) || true
else
  echo " FAIL run.py missing from repository root"
  ((FAIL++)) || true
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [[ $FAIL -gt 0 ]]; then
  echo "Fix the failures above before submitting."
  exit 1
fi
echo "All checks passed. Ready to zip for submission."
exit 0
