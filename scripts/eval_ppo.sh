#!/usr/bin/env bash
# Evaluate PPO baseline (no SLM).
# Output: results/ppo_results.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_ppo] Evaluating PPO baseline (200 episodes)"

python eval_ppo_slm.py --mode ppo --wandb-group fourrooms

echo "[eval_ppo] Done → results/ppo_results.json"
