#!/usr/bin/env bash
# Full pipeline from scratch — FourRooms + HigherLower.
# Designed to run unattended (Polyaxon, tmux, nohup).
#
#   FourRooms  : train (2M steps) → PPO eval → SLM eval → ASK eval → checkpoint ablation
#   HigherLower: train (~10min)   → PPO eval → SLM eval → ASK eval → checkpoint ablation
#
# Required environment variables:
#   WANDB_API_KEY   — Weights & Biases API key
#   HF_TOKEN        — HuggingFace token (for model download)
#
# Optional:
#   SKIP_INSTALL=1  — skip pip install (use if image already has deps)
#   SKIP_FR_TRAIN=1 — skip FourRooms training (checkpoint already exists)
#   SKIP_HL_TRAIN=1 — skip HigherLower training (checkpoint already exists)
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_all_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOGFILE") 2>&1

log() { echo ""; echo "[$(date '+%H:%M:%S')] $*"; }
log_banner() {
    echo ""
    echo "=========================================="
    echo "  $*"
    echo "  $(date)"
    echo "=========================================="
}

log_banner "ASK-POMDP — Full Pipeline"
echo "  Log: $LOGFILE"

# ── Environment ───────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="$(pwd)"
export TOKENIZERS_PARALLELISM=false

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ── Install dependencies ───────────────────────────────────────────────────────
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    log "Installing dependencies..."
    pip install -e ".[dev]" --quiet
    pip install popgym --quiet
fi

# ── Constants ─────────────────────────────────────────────────────────────────
FR_CKPT_DIR="runs/ppo/checkpoints"
HL_CKPT_DIR="runs/higher_lower/checkpoints"
FR_REWARDS=(010 030 050 070)
HL_REWARDS=(010 020 030 040)

SKIP_FR_TRAIN="${SKIP_FR_TRAIN:-0}"
SKIP_HL_TRAIN="${SKIP_HL_TRAIN:-0}"


# ══════════════════════════════════════════════════════════════════════════════
# FOURROOMS
# ══════════════════════════════════════════════════════════════════════════════
log_banner "FourRooms Pipeline"

# ── FR 1. Train ───────────────────────────────────────────────────────────────
if [ "$SKIP_FR_TRAIN" = "1" ]; then
    log "[FR 1/5] Skipping FourRooms training (SKIP_FR_TRAIN=1)"
else
    log "[FR 1/5] Training PPO on MiniGrid-FourRooms-v0 (2M steps)..."
    python train_ppo.py
    log "  Done -> runs/ppo/model.zip"
fi

# ── FR 2. PPO baseline ────────────────────────────────────────────────────────
log "[FR 2/5] PPO baseline eval..."
python eval_ppo_slm.py --mode ppo --wandb-group fourrooms
log "  Done -> results/ppo_results.json"

# ── FR 3. SLM-only ────────────────────────────────────────────────────────────
log "[FR 3/5] SLM-only baselines..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  -> SLM-only $SLM"
    python eval_ppo_slm.py --mode slm --slm "$SLM" --wandb-group fourrooms
done

# ── FR 4. ASK ─────────────────────────────────────────────────────────────────
log "[FR 4/5] ASK — Optuna + eval..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  -> ASK $SLM"
    python eval_ppo_slm.py --mode ask --slm "$SLM" --wandb-group fourrooms
done

# ── FR 5. Checkpoint ablation ─────────────────────────────────────────────────
log "[FR 5/5] Checkpoint ablation (PPO optimality)..."
for R in "${FR_REWARDS[@]}"; do
    MODEL="${FR_CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    log "  -- FourRooms checkpoint reward >= 0.${R}"

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

log_banner "FourRooms complete -> results/"


# ══════════════════════════════════════════════════════════════════════════════
# HIGHERLOWER
# ══════════════════════════════════════════════════════════════════════════════
log_banner "HigherLower Pipeline"

# ── HL 1. Train ───────────────────────────────────────────────────────────────
if [ "$SKIP_HL_TRAIN" = "1" ]; then
    log "[HL 1/5] Skipping HigherLower training (SKIP_HL_TRAIN=1)"
else
    log "[HL 1/5] Training PPO on HigherLower..."
    python higher_lower/train.py
    log "  Done -> runs/higher_lower/model.zip"
fi

# ── HL 2. PPO baseline ────────────────────────────────────────────────────────
log "[HL 2/5] PPO baseline eval..."
python higher_lower/eval.py --mode ppo --wandb-group higherlower
log "  Done -> higher_lower/results/ppo_results.json"

# ── HL 3. SLM-only ────────────────────────────────────────────────────────────
log "[HL 3/5] SLM-only baselines..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  -> SLM-only $SLM"
    python higher_lower/eval.py --mode slm --slm "$SLM" --wandb-group higherlower
done

# ── HL 4. ASK ─────────────────────────────────────────────────────────────────
log "[HL 4/5] ASK — Optuna + eval..."
for SLM in qwen3.5-2b qwen3.5-4b; do
    log "  -> ASK $SLM"
    python higher_lower/eval.py --mode ask --slm "$SLM" --wandb-group higherlower
done

# ── HL 5. Checkpoint ablation ─────────────────────────────────────────────────
log "[HL 5/5] Checkpoint ablation (PPO optimality)..."
for R in "${HL_REWARDS[@]}"; do
    MODEL="${HL_CKPT_DIR}/model_reward_${R}"
    TAG="ckpt_r${R}"
    log "  -- HigherLower checkpoint reward >= 0.${R}"

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

log_banner "All done! FourRooms -> results/  |  HigherLower -> higher_lower/results/"
