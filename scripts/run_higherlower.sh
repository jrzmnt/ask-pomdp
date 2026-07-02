#!/usr/bin/env bash
# Full HigherLower pipeline: train → eval → checkpoint ablation.
#
# Checkpoints are saved at reward thresholds [0.1, 0.2, 0.3, 0.4]
# (not fixed step intervals) so each snapshot is qualitatively different.
#
# W&B project : ask-pomdp-v2
# W&B groups  : higherlower  |  higherlower_checkpoints
#
# Estimated runtime on RTX 3060:
#   Train          : ~10min
#   Eval (SLM+ASK) : ~1.5h (2B) + ~2.5h (4B)
#   Checkpoints    : ~1.5h (4 ckpts × 2 models)
#   Total          : ~6h
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SKIP_TRAIN=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

CKPT_DIR="runs/higher_lower/checkpoints"
REWARDS=(010 020 030 040)

echo "=========================================="
echo "  HigherLower — Full Pipeline"
echo "  W&B project: ask-pomdp-v2"
[[ $SKIP_TRAIN -eq 1 ]] && echo "  (--skip-train: reusing existing model)"
echo "=========================================="

# ── 1. Train ──────────────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo ""
    echo "[1/5] Training PPO on HigherLower..."
    python higher_lower/train.py
    echo "  ✓ Model → runs/higher_lower/model.zip"
else
    echo ""
    echo "[1/5] Skipping training (--skip-train)"
fi

# ── 2. PPO baseline ───────────────────────────────────────────────────────────
echo ""
echo "[2/5] Evaluating PPO baseline..."
python higher_lower/eval.py --mode ppo --wandb-group higherlower
echo "  ✓ higher_lower/results/ppo_results.json"

# ── 3. SLM-only baselines ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Evaluating SLM-only baselines..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode slm --slm "$SLM" --wandb-group higherlower
done

# ── 4. ASK (Optuna + eval) ────────────────────────────────────────────────────
echo ""
echo "[4/5] ASK — Optuna τ tuning + evaluation..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    echo "  → $SLM"
    python higher_lower/eval.py --mode ask --slm "$SLM" --wandb-group higherlower
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

    python higher_lower/eval.py \
        --mode ppo \
        --model-path "${MODEL}" \
        --tag "${TAG}" \
        --wandb-group higherlower_checkpoints

    for SLM in qwen3.5-2b qwen3.5-4b; do
        python higher_lower/eval.py \
            --mode ask \
            --slm "${SLM}" \
            --model-path "${MODEL}" \
            --tag "${TAG}" \
            --n-optuna-trials 10 \
            --wandb-group higherlower_checkpoints
    done
done

echo ""
echo "=========================================="
echo "  HigherLower pipeline complete."
echo "  Results → higher_lower/results/"
echo "  W&B    → ask-pomdp-v2 / higherlower*"
echo "=========================================="
