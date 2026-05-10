#!/usr/bin/env bash
# Full pipeline for MiniGrid-FourRooms-v0.
# train → eval PPO → eval SLM (all models) → eval ASK (all models) → ablations
# Output: results/
set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo " FourRooms — Full Experiment Pipeline"
echo "========================================"

bash scripts/train.sh
bash scripts/eval_ppo.sh
bash scripts/eval_slm.sh
bash scripts/eval_slm_qwen3.sh
bash scripts/eval_ask.sh
bash scripts/eval_ask_qwen3.sh
bash scripts/ablation_threshold.sh
bash scripts/ablation_mc_samples.sh
bash scripts/ablation_always_ask.sh

echo ""
echo "========================================"
echo " All results saved to results/"
echo "========================================"
