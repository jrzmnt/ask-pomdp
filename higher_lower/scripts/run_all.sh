#!/usr/bin/env bash
# Full pipeline for HigherLower (POPGym).
# train → eval PPO → eval SLM (all models) → eval ASK (all models)
# Output: higher_lower/results/
# Estimated runtime: ~5–10h
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "========================================"
echo " HigherLower — Full Experiment Pipeline"
echo "========================================"

bash higher_lower/scripts/train.sh
bash higher_lower/scripts/eval_ppo.sh
bash higher_lower/scripts/eval_slm.sh
bash higher_lower/scripts/eval_ask.sh
bash higher_lower/scripts/ablation_threshold.sh
bash higher_lower/scripts/ablation_mc_samples.sh
bash higher_lower/scripts/ablation_always_ask.sh

echo "========================================"
echo " All done → higher_lower/results/"
echo "========================================"
