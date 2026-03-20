#!/usr/bin/env bash
set -euo pipefail

# Install bun
curl -fsSL https://bun.sh/install | bash
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# Install Python dependencies
pip install --upgrade pip
pip install -e ".[dev]"

echo "DevContainer ready — node $(node --version), bun $(bun --version), python $(python --version)"
