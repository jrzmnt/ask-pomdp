#!/usr/bin/env bash
# Evaluate SLM-only on HigherLower — Qwen3.5-2B and Qwen3.5-4B.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_eval_slm] Evaluating SLM-only..."

for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode slm --slm "$SLM" --wandb-group higherlower
done

echo "[hl_eval_slm] Done → higher_lower/results/slm_*_results.json"
