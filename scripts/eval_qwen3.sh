#!/usr/bin/env bash
# Full evaluation pipeline for Qwen3 models: SLM-only + ASK.
# Mirrors eval_slm.sh + eval_ask.sh but for Qwen3-0.8B and Qwen3-2B.
# Runtime: ~8–16h
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_qwen3] Starting full Qwen3 evaluation pipeline"

bash scripts/eval_slm_qwen3.sh
bash scripts/eval_ask_qwen3.sh

echo "[eval_qwen3] All done → results/slm_qwen3_*.json  results/ask_qwen3_*.json"
