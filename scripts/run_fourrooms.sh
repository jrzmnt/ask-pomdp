#!/usr/bin/env bash
# Full FourRooms pipeline: train → eval → checkpoint ablation.
#
# Checkpoints are saved at reward thresholds [0.1, 0.3, 0.5, 0.7]
# (not fixed step intervals) so each snapshot is qualitatively different.
#
# W&B project : ask-pomdp-v2
# W&B groups  : fourrooms  |  fourrooms_checkpoints
#
# Estimated runtime on RTX 3060:
#   Train          : ~3–5h
#   Eval (SLM+ASK) : ~3h (2B) + ~5h (4B)
#   Checkpoints    : ~2h (4 ckpts × 2 models)
#   Total          : ~13–15h
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SKIP_TRAIN=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

CKPT_DIR="runs/ppo/checkpoints"
REWARDS=(010 030 050 070)

echo "=========================================="
echo "  FourRooms — Full Pipeline"
echo "  W&B project: ask-pomdp-v2"
[[ $SKIP_TRAIN -eq 1 ]] && echo "  (--skip-train: reusing existing model)"
echo "=========================================="

# ── 1. Train ──────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    echo "[1/5] Training PPO on MiniGrid-FourRooms-v0..."
    python train_ppo.py
    echo "  ✓ Model → runs/ppo/model.zip"
else
    echo ""
    echo "[1/5] Skipping training (--skip-train)"
fi

# ── 2. PPO baseline ───────────────────────────────────────────────────────────
echo ""
echo "[2/5] Evaluating PPO baseline..."
python eval_ppo_slm.py --mode ppo --wandb-group fourrooms
echo "  ✓ results/ppo_results.json"

# ── 3. SLM-only baselines ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Evaluating SLM-only baselines..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python eval_ppo_slm.py --mode slm --slm "$SLM" --wandb-group fourrooms
done

# ── 4. ASK (Optuna + eval) ────────────────────────────────────────────────────
echo ""
echo "[4/5] ASK — Optuna τ tuning + evaluation..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python eval_ppo_slm.py --mode ask --slm "$SLM" --wandb-group fourrooms
done

# ── 5. Checkpoint ablation (PPO optimality) ───────────────────────────────────
echo ""
echo "[5/5] Checkpoint ablation (reward thresholds: ${REWARDS[*]})..."

for R in "${REWARDS[@]}"; do
    MODEL="${CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    echo ""
    echo "  ── Checkpoint reward ≥ 0.${R}"

    if [ ! -f "${MODEL}.zip" ]; then
        echo "  [SKIP] ${MODEL}.zip not found"
        continue
    fi

    python eval_ppo_slm.py \
        --mode ppo \
        --model-path "${MODEL}" \
        --tag "${TAG}" \
        --wandb-group fourrooms_checkpoints

    for SLM in qwen3.5-2b qwen3.5-4b; do
        python eval_ppo_slm.py \
            --mode ask \
            --slm "${SLM}" \
            --model-path "${MODEL}" \
            --tag "${TAG}" \
            --n-optuna-trials 10 \
            --wandb-group fourrooms_checkpoints
    done
done

echo ""
echo "=========================================="
echo "  FourRooms pipeline complete."
echo "  Results → results/"
echo "  W&B    → ask-pomdp-v2 / fourrooms*"
echo "=========================================="
