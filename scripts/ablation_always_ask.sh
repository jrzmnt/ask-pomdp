#!/usr/bin/env bash
# Ablation: always ask (τ = 0, IR ≈ 100%).
#
# Demonstrates that indiscriminate SLM querying is suboptimal
# compared to uncertainty-gated querying (optimal τ from Optuna).
# Complements the PPO-only baseline (τ = ∞, IR = 0%).
#
# Output: results/ask_qwen_{0.5b,1.5b}_results_always_ask.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[ablation_always_ask] Evaluating ASK with τ=0 (always query SLM)"

python eval_ppo_slm.py \
    --mode ask \
    --slm 0.5b \
    --threshold 0.0 \
    --tag "always_ask" \
    --wandb-group ablation_always_ask

python eval_ppo_slm.py \
    --mode ask \
    --slm 1.5b \
    --threshold 0.0 \
    --tag "always_ask" \
    --wandb-group ablation_always_ask

echo ""
echo "[ablation_always_ask] Done → results/ask_qwen_*_results_always_ask.json"
