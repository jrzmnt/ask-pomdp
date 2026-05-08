#!/usr/bin/env bash
# Ablation: threshold τ sensitivity.
#
# Sweeps τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}
# with fixed τ (no Optuna) to show the reward-vs-threshold curve.
# Uses Qwen 1.5B as the representative model.
#
# Output: results/ask_qwen_1.5b_results_threshold_*.json
#         results/ask_qwen_1.5b_episodes_threshold_*.csv
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[ablation_threshold] Sweeping τ for Qwen 1.5B"

for TAU in 0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4 1.6 1.8 2.0; do
    TAU_TAG="threshold_${TAU/./}"   # e.g. "threshold_08" for 0.8
    echo ""
    echo "  τ = ${TAU}"
    python eval_ppo_slm.py \
        --mode ask \
        --slm 1.5b \
        --threshold "${TAU}" \
        --tag "${TAU_TAG}" \
        --wandb-group ablation_threshold
done

echo ""
echo "[ablation_threshold] Done → results/ask_qwen_1.5b_results_threshold_*.json"
