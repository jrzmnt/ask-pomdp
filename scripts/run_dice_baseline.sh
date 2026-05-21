#!/usr/bin/env bash
# Dice baseline: re-run the ASK pipeline with the SLM replaced by uniform random
# action sampling (--slm random). Same PPO, same MC-dropout gate, same Optuna
# τ search — only the "consultant" is swapped, so any gap vs. real-SLM ASK is
# attributable to the SLM's decisions (not just the gate firing on uncertain
# states).
#
# Usage from repo root:
#   bash scripts/run_dice_baseline.sh                 # all three envs, ASK + SLM-only
#   ENVS="fourrooms doorkey" bash scripts/run_dice_baseline.sh
#   N_EP=25 N_OPTUNA=5 bash scripts/run_dice_baseline.sh
#   DOORKEY_SIZE=8 bash scripts/run_dice_baseline.sh
#
# Outputs land alongside the real-SLM JSONs with tag "random":
#   results/{slm,ask}_random_*.json                     (FourRooms)
#   higher_lower/results/{slm,ask}_random_*.json
#   door_key/results/{slm,ask}_random_*.json

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

N_EP="${N_EP:-100}"
N_EVAL_EP="${N_EVAL_EP:-100}"
N_OPTUNA="${N_OPTUNA:-10}"
GROUP_SUFFIX="${GROUP_SUFFIX:-dice}"
ENVS="${ENVS:-fourrooms higherlower doorkey}"
DOORKEY_SIZE="${DOORKEY_SIZE:-8}"

run_fr () {
    echo "=== FourRooms — dice (SLM-only) ==="
    python eval_ppo_slm.py --mode slm --slm random \
        --n-episodes "$N_EP" --wandb-group "fourrooms_${GROUP_SUFFIX}"

    echo "=== FourRooms — dice (ASK) ==="
    python eval_ppo_slm.py --mode ask --slm random \
        --n-episodes "$N_EP" --n-eval-episodes "$N_EVAL_EP" \
        --n-optuna-trials "$N_OPTUNA" --wandb-group "fourrooms_${GROUP_SUFFIX}"
}

run_hl () {
    echo "=== HigherLower — dice (SLM-only) ==="
    python higher_lower/eval.py --mode slm --slm random \
        --n-episodes "$N_EP" --wandb-group "higherlower_${GROUP_SUFFIX}"

    echo "=== HigherLower — dice (ASK) ==="
    python higher_lower/eval.py --mode ask --slm random \
        --n-episodes "$N_EP" --n-eval-episodes "$N_EVAL_EP" \
        --n-optuna-trials "$N_OPTUNA" --wandb-group "higherlower_${GROUP_SUFFIX}"
}

run_dk () {
    echo "=== DoorKey-${DOORKEY_SIZE} — dice (SLM-only) ==="
    python door_key/eval.py --mode slm --slm random --size "$DOORKEY_SIZE" \
        --n-episodes "$N_EP" --wandb-group "doorkey_${GROUP_SUFFIX}"

    echo "=== DoorKey-${DOORKEY_SIZE} — dice (ASK) ==="
    python door_key/eval.py --mode ask --slm random --size "$DOORKEY_SIZE" \
        --n-episodes "$N_EP" --n-eval-episodes "$N_EVAL_EP" \
        --n-optuna-trials "$N_OPTUNA" --wandb-group "doorkey_${GROUP_SUFFIX}"
}

for env in $ENVS; do
    case "$env" in
        fourrooms|fr)         run_fr ;;
        higherlower|hl)       run_hl ;;
        doorkey|dk)           run_dk ;;
        *) echo "[warn] unknown env: $env (expected fourrooms|higherlower|doorkey)" >&2 ;;
    esac
done

echo "Done. Tag = random; results/ + higher_lower/results/ + door_key/results/"
