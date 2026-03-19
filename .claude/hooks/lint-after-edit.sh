#!/usr/bin/env bash
set -uo pipefail
# Run ruff on Python files after edit/write.
# Claude Code hook: reads JSON from stdin, non-blocking (exit 0).

command -v jq >/dev/null 2>&1 || { echo "Warning: jq not found, skipping lint hook" >&2; exit 0; }

INPUT=$(cat)
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path')

# Only lint Python files
if [[ "$FILE_PATH" == *.py ]]; then
  ruff check -- "$FILE_PATH" 2>&1 | tail -5
fi

exit 0
