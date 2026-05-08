#!/usr/bin/env bash
# Full pipeline: train → eval PPO → eval SLM → eval ASK.
# Run this script to reproduce all paper results from scratch.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo " ASK-POMDP — Full Experiment Pipeline"
echo "========================================"
echo ""

bash scripts/train.sh
bash scripts/eval_ppo.sh
bash scripts/eval_slm.sh
bash scripts/eval_ask.sh

echo ""
echo "========================================"
echo " All results saved to results/"
echo "========================================"
