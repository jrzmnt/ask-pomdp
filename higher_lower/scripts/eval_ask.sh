#!/usr/bin/env bash
# Tune τ via Optuna and evaluate ASK on HigherLower — all four models (~4–8h)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_eval_ask] Optuna + eval ASK (all models)..."
for SLM in 0.5b 1.5b qwen3-0.6b qwen3-1.7b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode ask --slm "$SLM"
done
echo "[hl_eval_ask] Done → higher_lower/results/ask_*_results.json"
