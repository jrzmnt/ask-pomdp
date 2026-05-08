#!/usr/bin/env bash
# Evaluate SLM-only baselines (Qwen 0.5B and 1.5B).
# Output: results/slm_qwen_{0.5b,1.5b}_results.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_slm] Evaluating SLM-only baselines (200 episodes each)"

python eval_ppo_slm.py --mode slm --slm 0.5b --wandb-group main
python eval_ppo_slm.py --mode slm --slm 1.5b --wandb-group main

echo "[eval_slm] Done → results/slm_qwen_*.json"
