#!/usr/bin/env bash
# Full DoorKey pipeline: train → eval PPO → eval SLM → eval ASK, for all sizes.
#
# Usage:
#   bash door_key/scripts/run_doorkey.sh              # all sizes, 500k steps each
#   bash door_key/scripts/run_doorkey.sh --sizes 5 6  # specific sizes only
#   bash door_key/scripts/run_doorkey.sh --timesteps 1000000
#
# Wall time estimates (CPU):
#   5×5:  ~30 min train + ~20 min eval = ~50 min
#   6×6:  ~45 min train + ~25 min eval = ~70 min
#   8×8:  ~1.5h  train + ~40 min eval  = ~2.2h
#  16×16: ~4h    train + ~2h   eval    = ~6h
# ----------------------------------------------------------

set -euo pipefail

SIZES=(5 6 8 16)
TIMESTEPS=500000
SLM="qwen3.5-2b"

# Parse custom --sizes and --timesteps flags (rest passed through to scripts)
PASSTHROUGH=()
i=0
ARGS=("$@")
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --sizes)
            SIZES=()
            ((i++))
            while [[ $i -lt ${#ARGS[@]} && "${ARGS[$i]}" =~ ^[0-9]+$ ]]; do
                SIZES+=("${ARGS[$i]}")
                ((i++))
            done
            ;;
        --timesteps)
            ((i++))
            TIMESTEPS="${ARGS[$i]}"
            ((i++))
            ;;
        *)
            PASSTHROUGH+=("${ARGS[$i]}")
            ((i++))
            ;;
    esac
done

echo "========================================"
echo "  ASK-POMDP — DoorKey Full Pipeline"
echo "  Sizes: ${SIZES[*]}"
echo "  Timesteps per size: ${TIMESTEPS}"
echo "  SLM: ${SLM}"
echo "========================================"

for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "──────────────────────────────────────"
    echo "  DoorKey-${SIZE}x${SIZE}"
    echo "──────────────────────────────────────"

    echo "[1/3] Training PPO for ${SIZE}x${SIZE}..."
    python door_key/train.py --size "$SIZE" --timesteps "$TIMESTEPS"

    echo "[2/3] Evaluating PPO for ${SIZE}x${SIZE}..."
    python door_key/eval.py --mode ppo --size "$SIZE"

    echo "[3/3] Evaluating SLM-only for ${SIZE}x${SIZE}..."
    python door_key/eval.py --mode slm --slm "$SLM" --size "$SIZE"

    echo "[4/4] Evaluating ASK (Optuna τ search) for ${SIZE}x${SIZE}..."
    python door_key/eval.py --mode ask --slm "$SLM" --size "$SIZE"

    echo "  ✓ DoorKey-${SIZE}x${SIZE} complete"
done

echo ""
echo "========================================"
echo "  All DoorKey sizes complete."
echo "  Results → door_key/results/"
echo "  Thresholds → door_key/results/thresholds.json"
echo "========================================"
