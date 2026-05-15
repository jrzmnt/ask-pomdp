#!/usr/bin/env bash
# HigherLower prompt ablation: basic / enriched / stateful / stateful+rationale
# (25 episodes on Qwen3.5-2B by default). Run from repo root.
#
# Optional env: N_EP (default 25), SLM (default qwen3.5-2b), GROUP, WANDB_MODE

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$(pwd)"

N_EP="${N_EP:-25}"
SLM="${SLM:-qwen3.5-2b}"
GROUP="${GROUP:-higherlower_prompt_ablation}"

run_hl () {
  local tag="$1"
  shift
  echo "=== tag=$tag $* ==="
  python higher_lower/eval.py --mode slm --slm "$SLM" --wandb-group "$GROUP" \
    --n-episodes "$N_EP" --tag "$tag" "$@"
}

run_hl basic --prompt-style basic
run_hl enriched --prompt-style enriched
run_hl stateful --prompt-style stateful
run_hl stateful_rat --prompt-style stateful --prompt-rationale

echo "Done. Results → higher_lower/results/slm_*_episodes_<tag>.csv  (W&B group $GROUP)"
