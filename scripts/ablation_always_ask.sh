#!/usr/bin/env bash
# Ablation: always ask (τ = 0, IR = 100%) — FourRooms.
#
# Demonstrates that indiscriminate SLM querying is suboptimal
# compared to uncertainty-gated querying (optimal τ from Optuna).
#
# Output: results/ask_*_results_always_ask.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "[ablation_always_ask] Evaluating ASK with τ=0 (always query SLM)"

for SLM in qwen3.5-2b qwen3.5-4b; do
    echo ""
    echo "  Model: ${SLM}"
    python eval_ppo_slm.py \
        --mode ask \
        --slm "${SLM}" \
        --threshold 0.0 \
        --tag "always_ask" \
        --wandb-group fourrooms_ablation_always_ask
done

echo ""
echo "[ablation_always_ask] Done → results/ask_*_results_always_ask.json"
