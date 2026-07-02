#!/usr/bin/env bash
# Ablation: threshold τ sensitivity — Qwen3.5-2B and Qwen3.5-4B.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[ablation_threshold] Sweeping τ"

for SLM in qwen3.5-2b qwen3.5-4b; do
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
