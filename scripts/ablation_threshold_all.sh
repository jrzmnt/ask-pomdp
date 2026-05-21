#!/usr/bin/env bash
# Dense τ sweep for Fig 1 across all three environments. Runs ASK at a grid of
# fixed τ values (no Optuna) on the FULL PPO model so reward and intervention
# rate are measured under a single, fixed policy. Produces tagged JSON+CSV
# files that plots/make_figures.py picks up:
#
#   results/ask_{tag}_results_threshold_{TAU/./}.json
#   higher_lower/results/ask_{tag}_results_threshold_{TAU/./}.json
#   door_key/results/ask_{tag}_results_s{SIZE}_threshold_{TAU/./}.json
#
# Usage:
#   bash scripts/ablation_threshold_all.sh
#   ENVS="fourrooms doorkey" bash scripts/ablation_threshold_all.sh
#   SLMS="qwen3.5-2b" bash scripts/ablation_threshold_all.sh
#   FR_TAUS="0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4 1.6 1.8 2.0" N_EP=50 bash ...
#   DK_TAUS="0.5 1.0 1.2 1.4 1.6 1.8 2.0" DOORKEY_SIZE=8 bash ...

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-offline}"

# Reasonable defaults per env (informed by current Optuna best τ).
ENVS="${ENVS:-fourrooms higherlower doorkey}"
SLMS="${SLMS:-qwen3.5-2b qwen3.5-4b}"
N_EP="${N_EP:-100}"
DOORKEY_SIZE="${DOORKEY_SIZE:-8}"
GROUP="${GROUP:-ablation_threshold}"

FR_TAUS="${FR_TAUS:-0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.3 1.7}"   # full PPO τ* ≈ 0.46–0.48
HL_TAUS="${HL_TAUS:-0.005 0.02 0.05 0.1 0.2 0.4 0.6 0.9 1.2 1.5}"     # full PPO τ* ≈ 0.013
DK_TAUS="${DK_TAUS:-0.3 0.6 0.9 1.1 1.3 1.5 1.7 1.9}"                  # full PPO τ* ≈ 1.45–1.53

run_fr () {
    local SLM="$1"
    for TAU in $FR_TAUS; do
        local TAG="threshold_${TAU/./}"
        echo "=== FourRooms ${SLM} τ=${TAU} ==="
        python eval_ppo_slm.py --mode ask --slm "$SLM" \
            --threshold "$TAU" --n-episodes "$N_EP" \
            --tag "$TAG" --wandb-group "$GROUP"
    done
}

run_hl () {
    local SLM="$1"
    for TAU in $HL_TAUS; do
        local TAG="threshold_${TAU/./}"
        echo "=== HigherLower ${SLM} τ=${TAU} ==="
        python higher_lower/eval.py --mode ask --slm "$SLM" \
            --threshold "$TAU" --n-episodes "$N_EP" \
            --tag "$TAG" --wandb-group "$GROUP"
    done
}

run_dk () {
    local SLM="$1"
    for TAU in $DK_TAUS; do
        local TAG="threshold_${TAU/./}"
        echo "=== DoorKey-${DOORKEY_SIZE} ${SLM} τ=${TAU} ==="
        python door_key/eval.py --mode ask --slm "$SLM" --size "$DOORKEY_SIZE" \
            --threshold "$TAU" --n-episodes "$N_EP" \
            --tag "$TAG" --wandb-group "$GROUP"
    done
}

for SLM in $SLMS; do
    for env in $ENVS; do
        case "$env" in
            fourrooms|fr)   run_fr "$SLM" ;;
            higherlower|hl) run_hl "$SLM" ;;
            doorkey|dk)     run_dk "$SLM" ;;
            *) echo "[warn] unknown env '$env'" >&2 ;;
        esac
    done
done

echo ""
echo "Done. Replot Fig 1 with:  python plots/make_figures.py fig1"
