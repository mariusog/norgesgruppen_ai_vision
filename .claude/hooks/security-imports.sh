#!/usr/bin/env bash
# Blocks forbidden imports that will disqualify the competition submission.
# Runs as a PostToolUse hook after Edit|Write on Python files.
#
# Scans for actual import STATEMENTS only (lines starting with import/from),
# not string literals or comments that happen to contain these words.

set -euo pipefail

# Read the tool input JSON from stdin
INPUT=$(cat)

# Extract the file path from the tool input
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
tool_input = data.get('tool_input', data)
print(tool_input.get('file_path', ''))
" 2>/dev/null || echo "")

# Only check Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Only check if the file exists
if [[ -z "$FILE_PATH" ]] || [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

# Match actual import statements only — anchored at start of line (with optional leading whitespace)
# These patterns match: import os, import os.path, from os import ..., etc.
# They do NOT match occurrences inside string literals or comments.
SINGLE_PATTERN='^\s*(import (os|subprocess|socket)\b|from (os|subprocess|socket)(\s|\.|;|$))'

MATCHES=$(grep -Pn "$SINGLE_PATTERN" "$FILE_PATH" 2>/dev/null || true)

if [[ -n "$MATCHES" ]]; then
  echo ""
  echo "COMPETITION SECURITY VIOLATION: $FILE_PATH"
  echo "The sandbox security scanner will DISQUALIFY submissions using these imports:"
  echo "  import os | import subprocess | import socket"
  echo ""
  echo "Violations found:"
  echo "$MATCHES" | head -5 | sed 's/^/  /'
  echo ""
  echo "Fix: use pathlib.Path for file ops; avoid subprocess/socket entirely."
  echo ""
  exit 2
fi

exit 0
