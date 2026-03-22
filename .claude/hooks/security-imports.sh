#!/usr/bin/env bash
# Blocks forbidden imports/calls that will disqualify the competition submission.
# Runs as a PostToolUse hook after Edit|Write on Python files.
#
# Full blocked list per competition security scanner:
#   Modules: os, subprocess, socket, ctypes, builtins
#   Builtins: eval(), exec(), compile(), __import__()

set -euo pipefail

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
tool_input = data.get('tool_input', data)
print(tool_input.get('file_path', ''))
" 2>/dev/null || echo "")

if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

if [[ -z "$FILE_PATH" ]] || [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

# Forbidden import statements (anchored to start of line, with optional whitespace)
IMPORT_PATTERN='^\s*(import (os|subprocess|socket|ctypes|builtins)\b|from (os|subprocess|socket|ctypes|builtins)(\s|\.|;|$))'

# Forbidden builtin calls
CALL_PATTERN='^\s*(eval|exec|compile|__import__)\s*\('

IMPORT_MATCHES=$(grep -Pn "$IMPORT_PATTERN" "$FILE_PATH" 2>/dev/null || true)
CALL_MATCHES=$(grep -Pn "$CALL_PATTERN" "$FILE_PATH" 2>/dev/null || true)

if [[ -n "$IMPORT_MATCHES" ]] || [[ -n "$CALL_MATCHES" ]]; then
  echo ""
  echo "COMPETITION SECURITY VIOLATION: $FILE_PATH"
  echo "Sandbox scanner DISQUALIFIES submissions using these:"
  echo "  Blocked modules: os, subprocess, socket, ctypes, builtins"
  echo "  Blocked builtins: eval(), exec(), compile(), __import__()"
  echo ""
  if [[ -n "$IMPORT_MATCHES" ]]; then
    echo "Forbidden imports:"
    echo "$IMPORT_MATCHES" | head -5 | sed 's/^/  /'
  fi
  if [[ -n "$CALL_MATCHES" ]]; then
    echo "Forbidden calls:"
    echo "$CALL_MATCHES" | head -5 | sed 's/^/  /'
  fi
  echo ""
  echo "Fix: use pathlib.Path for file ops; avoid subprocess/socket/ctypes entirely."
  exit 2
fi

exit 0
