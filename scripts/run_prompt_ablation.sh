#!/usr/bin/env bash
# Run the 25-episode prompt ablation matrix (Qwen3.5-2B) from the SLM prompt enrichment plan.
# Usage: from repo root:
#   bash scripts/run_prompt_ablation.sh
#
# Optional env: N_EP (default 25), SLM (default qwen3.5-2b), GROUP, WANDB_MODE
# Requires HF model access; GPU recommended.

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

N_EP="${N_EP:-25}"
SLM="${SLM:-qwen3.5-2b}"
GROUP="${GROUP:-fourrooms_prompt_ablation}"

run_slm () {
  local tag="$1"
  shift
  echo "=== tag=$tag $* ==="
  python eval_ppo_slm.py --mode slm --slm "$SLM" --wandb-group "$GROUP" \
    --n-episodes "$N_EP" --tag "$tag" "$@"
}

run_slm basic --prompt-style basic
run_slm enriched --prompt-style enriched
run_slm stateful_min --prompt-style stateful_min
run_slm stateful --prompt-style stateful
run_slm stateful_rat --prompt-style stateful --prompt-rationale

echo "Done. Results under results/slm_*_episodes_<tag>.csv and W&B group $GROUP"
