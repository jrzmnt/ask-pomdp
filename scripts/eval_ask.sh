#!/usr/bin/env bash
# Tune τ via Optuna and evaluate ASK — Qwen3.5-2B and Qwen3.5-4B.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_ask] Tuning τ (15 Optuna trials) and evaluating ASK"

for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python eval_ppo_slm.py --mode ask --slm "$SLM" --wandb-group fourrooms
done

echo "[eval_ask] Done → results/ask_*.json"
