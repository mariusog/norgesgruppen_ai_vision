#!/usr/bin/env bash
set -uo pipefail
# Block git commit if tests fail.
# Claude Code hook: reads JSON from stdin, exit 2 to block.

command -v jq >/dev/null 2>&1 || { echo "Error: jq is required for Claude Code hooks" >&2; exit 1; }

INPUT=$(cat)
COMMAND=$(printf '%s' "$INPUT" | jq -r '.tool_input.command')

# Only trigger on git commit commands
if [[ "$COMMAND" == *"git commit"* ]]; then
  RESULT=$(python -m pytest tests/ -q --tb=line -m 'not slow' 2>&1 | tail -5)
  EXIT=$?
  if [ $EXIT -ne 0 ]; then
    echo "BLOCKED: Tests must pass before committing." >&2
    echo "$RESULT" >&2
    exit 2
  fi
fi

exit 0
