#!/usr/bin/env bash
# Ablation: threshold τ sensitivity.
#
# Sweeps τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}
# with fixed τ (no Optuna) using one representative per model family:
#   - Qwen2.5-1.5B
#   - Qwen3-1.7B
#
# Output: results/ask_qwen25_1.5b_results_threshold_*.json
#         results/ask_qwen3_1.7b_results_threshold_*.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[ablation_threshold] Sweeping τ for Qwen2.5-1.5B and Qwen3-1.7B"

for SLM in 1.5b qwen3-1.7b; do
    echo ""
    echo "=== Model: ${SLM} ==="
    for TAU in 0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4 1.6 1.8 2.0; do
        TAU_TAG="threshold_${TAU/./}"
        echo "  τ = ${TAU}"
        python eval_ppo_slm.py \
            --mode ask \
            --slm "${SLM}" \
            --threshold "${TAU}" \
            --tag "${TAU_TAG}" \
            --wandb-group ablation_threshold
    done
done

echo ""
echo "[ablation_threshold] Done → results/ask_*_results_threshold_*.json"
