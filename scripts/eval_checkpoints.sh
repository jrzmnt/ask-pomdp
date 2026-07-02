#!/usr/bin/env bash
# Ablation: PPO optimality effect on ASK — FourRooms.
# Evaluates PPO + ASK for each reward-threshold checkpoint.
#
# Checkpoints saved by RewardThresholdCheckpointCallback:
#   model_reward_010.zip  (reward ≥ 0.10)
#   model_reward_030.zip  (reward ≥ 0.30)
#   model_reward_050.zip  (reward ≥ 0.50)
#   model_reward_070.zip  (reward ≥ 0.70)
#
# Output: results/
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CKPT_DIR="runs/ppo/checkpoints"
REWARDS=(010 030 050 070)

echo "[eval_checkpoints] PPO optimality ablation — FourRooms"
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
    python eval_ppo_slm.py \
        --mode ppo \
        --model-path "${MODEL}" \
        --tag "${TAG}" \
        --wandb-group fourrooms_checkpoints

    for SLM in qwen3.5-2b qwen3.5-4b; do
        echo "  → ASK eval (${SLM})"
        CUDA_VISIBLE_DEVICES=4  python eval_ppo_slm.py \
            --mode ask \
            --slm "${SLM}" \
            --model-path "${MODEL}" \
            --tag "${TAG}" \
            --n-optuna-trials 10 \
            --wandb-group fourrooms_checkpoints \
            --prompt-style stateful \
            --prompt-rationale
    done
    echo ""
done

echo "[eval_checkpoints] Done → results/"
