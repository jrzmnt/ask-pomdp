#!/usr/bin/env bash
# Create virtual environment and install all dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

uv venv --python 3.11
source .venv/bin/activate

uv pip install -e ".[dev]"

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
