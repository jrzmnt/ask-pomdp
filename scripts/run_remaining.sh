#!/usr/bin/env bash
# Roda tudo que falta:
#   FourRooms   — SLM-only, ASK, checkpoint ablation (PPO e checkpoints já existem)
#   HigherLower — pipeline completo (train + eval + checkpoint ablation)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

log() { echo ""; echo "[$(date '+%H:%M:%S')] $*"; }

echo "=========================================="
echo "  ASK-POMDP — Rodando o que falta"
echo "  $(date)"
echo "=========================================="

# ══════════════════════════════════════════════════════════════════════════════
# FOURROOMS — PPO já treinado e avaliado (results/ppo_results.json existe)
# ══════════════════════════════════════════════════════════════════════════════

log "[FR 1/3] SLM-only (2B e 4B)..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  → SLM-only $SLM"
    python eval_ppo_slm.py --mode slm --slm "$SLM" --wandb-group fourrooms
done

log "[FR 2/3] ASK — Optuna + eval (2B e 4B)..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  → ASK $SLM"
    python eval_ppo_slm.py --mode ask --slm "$SLM" --wandb-group fourrooms
done

log "[FR 3/3] Checkpoint ablation (PPO optimality)..."
FR_CKPT_DIR="runs/ppo/checkpoints"
for R in 010 030 050 070; do
    MODEL="${FR_CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    log "  ── FourRooms checkpoint reward >= 0.${R}"

    if [ ! -f "${MODEL}.zip" ]; then
        echo "  [SKIP] ${MODEL}.zip não encontrado"
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
echo "  FourRooms completo."
echo "=========================================="

# ══════════════════════════════════════════════════════════════════════════════
# HIGHERLOWER — pipeline completo
# ══════════════════════════════════════════════════════════════════════════════

log "[HL 1/5] Treinando PPO no HigherLower..."
python higher_lower/train.py
log "  ✓ Modelo → runs/higher_lower/model.zip"

log "[HL 2/5] PPO baseline..."
python higher_lower/eval.py --mode ppo --wandb-group higherlower
log "  ✓ higher_lower/results/ppo_results.json"

log "[HL 3/5] SLM-only (2B e 4B)..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  → SLM-only $SLM"
    python higher_lower/eval.py --mode slm --slm "$SLM" --wandb-group higherlower
done

log "[HL 4/5] ASK — Optuna + eval (2B e 4B)..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  → ASK $SLM"
    python higher_lower/eval.py --mode ask --slm "$SLM" --wandb-group higherlower
done

log "[HL 5/5] Checkpoint ablation (PPO optimality)..."
HL_CKPT_DIR="runs/higher_lower/checkpoints"
for R in 010 020 030 040; do
    MODEL="${HL_CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    log "  ── HigherLower checkpoint reward >= 0.${R}"

    if [ ! -f "${MODEL}.zip" ]; then
        echo "  [SKIP] ${MODEL}.zip não encontrado"
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
echo "  Tudo concluído!"
echo "  FourRooms   → results/"
echo "  HigherLower → higher_lower/results/"
echo "  W&B         → ask-pomdp-v2"
echo "  $(date)"
echo "=========================================="
