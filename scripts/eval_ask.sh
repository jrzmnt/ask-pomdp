#!/usr/bin/env bash
# Tune threshold τ via Optuna and evaluate ASK (PPO + SLM gated).
# Runs for both Qwen 0.5B and 1.5B.
# Output: results/ask_qwen_{0.5b,1.5b}_results.json
#         results/ask_qwen_{0.5b,1.5b}_episodes.csv
#         optuna.db
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_ask] Tuning τ (15 Optuna trials) and evaluating ASK"

python eval_ppo_slm.py --mode ask --slm 0.5b --wandb-group main
python eval_ppo_slm.py --mode ask --slm 1.5b --wandb-group main

echo "[eval_ask] Done → results/ask_qwen_*.json"
