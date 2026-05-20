#!/usr/bin/env bash
# Prompt ablation matrix for DoorKey (Qwen3.5-2B by default).
# Usage from repo root:
#   bash scripts/run_dk_prompt_ablation.sh
#
# Optional env: N_EP (default 25), SLM (default qwen3.5-2b),
#               SIZE (default 5), GROUP, WANDB_MODE
# Requires HF model access; GPU recommended.

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

N_EP="${N_EP:-25}"
SLM="${SLM:-qwen3.5-2b}"
SIZE="${SIZE:-5}"
GROUP="${GROUP:-doorkey_prompt_ablation}"

run_slm () {
  local tag="$1"
  shift
  echo "=== tag=$tag $* ==="
  python door_key/eval.py --mode slm --slm "$SLM" --size "$SIZE" \
    --wandb-group "$GROUP" --n-episodes "$N_EP" --tag "$tag" "$@"
}

run_slm basic        --prompt-style basic
run_slm enriched     --prompt-style enriched
run_slm stateful_min --prompt-style stateful_min
run_slm stateful     --prompt-style stateful
run_slm stateful_rat --prompt-style stateful --prompt-rationale

echo "Done. Results under door_key/results/slm_*_episodes_s${SIZE}_<tag>.csv and W&B group $GROUP"
