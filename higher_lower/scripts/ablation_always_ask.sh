#!/usr/bin/env bash
# Ablation: always ask (τ = 0, IR = 100%) for HigherLower.
#
# Demonstrates that indiscriminate SLM querying is suboptimal
# compared to uncertainty-gated querying (optimal τ from Optuna).
# Runs for all four models.
#
# Output: higher_lower/results/ask_*_results_always_ask.json
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "[hl_ablation_always_ask] Evaluating ASK with τ=0 (always query SLM)"

for SLM in 0.5b 1.5b qwen3-0.6b qwen3-1.7b; do
    echo ""
    echo "  Model: ${SLM}"
    python higher_lower/eval.py \
        --mode ask \
        --slm "${SLM}" \
        --threshold 0.0 \
        --tag "always_ask" \
        --wandb-group hl_ablation_always_ask
done

echo ""
echo "[hl_ablation_always_ask] Done → higher_lower/results/ask_*_results_always_ask.json"
