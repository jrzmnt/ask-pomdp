#!/usr/bin/env bash
# Evaluate PPO baseline on HigherLower (~1 min)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_eval_ppo] Evaluating PPO..."
python higher_lower/eval.py --mode ppo
echo "[hl_eval_ppo] Done → higher_lower/results/ppo_results.json"
