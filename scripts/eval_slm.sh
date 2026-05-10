#!/usr/bin/env bash
# Evaluate SLM-only baseline — Qwen3.5-2B and Qwen3.5-4B.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_slm] Evaluating SLM-only baselines (200 episodes each)"

for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python eval_ppo_slm.py --mode slm --slm "$SLM" --wandb-group fourrooms
done

echo "[eval_slm] Done → results/slm_*.json"
