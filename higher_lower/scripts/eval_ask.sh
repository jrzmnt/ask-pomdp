#!/usr/bin/env bash
# Tune τ via Optuna and evaluate ASK on HigherLower — Qwen3.5-2B and Qwen3.5-4B.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_eval_ask] Optuna + eval ASK..."

for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode ask --slm "$SLM" --wandb-group higherlower
done

echo "[hl_eval_ask] Done → higher_lower/results/ask_*_results.json"
