#!/usr/bin/env bash
# Evaluate SLM-only on HigherLower — all four models (~2–4h)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_eval_slm] Evaluating SLM-only (all models)..."
for SLM in 0.5b 1.5b qwen3-0.6b qwen3-1.7b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode slm --slm "$SLM"
done
echo "[hl_eval_slm] Done → higher_lower/results/slm_*_results.json"
