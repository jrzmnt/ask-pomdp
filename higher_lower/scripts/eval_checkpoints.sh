#!/usr/bin/env bash
# Ablation: PPO optimality effect on ASK — HigherLower.
# Evaluates PPO + ASK for each reward-threshold checkpoint.
#
# Checkpoints saved by RewardThresholdCheckpointCallback:
#   model_reward_010.zip  (reward ≥ 0.10)
#   model_reward_020.zip  (reward ≥ 0.20)
#   model_reward_030.zip  (reward ≥ 0.30)
#   model_reward_040.zip  (reward ≥ 0.40)
#
# Output: higher_lower/results/
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

CKPT_DIR="runs/higher_lower/checkpoints"
REWARDS=(010 020 030 040)

echo "[hl_eval_checkpoints] PPO optimality ablation — HigherLower"
echo "  Reward thresholds: ${REWARDS[*]}"
echo ""

for R in "${REWARDS[@]}"; do
    MODEL="${CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    LABEL="0.${R}"
    echo "=========================================="
    echo "  Checkpoint: reward ≥ ${LABEL}"
    echo "=========================================="

    if [ ! -f "${MODEL}.zip" ]; then
        echo "  [SKIP] ${MODEL}.zip not found"
        continue
    fi

    echo "  → PPO eval"
    python higher_lower/eval.py \
        --mode ppo \
        --model-path "${MODEL}" \
        --tag "${TAG}" \
        --wandb-group higherlower_checkpoints


    for SLM in qwen3.5-2b qwen3.5-4b; do
        echo "  → ASK eval (${SLM})"
        python higher_lower/eval.py \
            --mode ask \
            --slm "${SLM}" \
            --model-path "${MODEL}" \
            --tag "${TAG}" \
            --n-optuna-trials 10 \
            --wandb-group higherlower_checkpoints \
            --prompt-style stateful \
            --prompt-rationale
    done
    echo ""
done

echo "[hl_eval_checkpoints] Done → higher_lower/results/"
