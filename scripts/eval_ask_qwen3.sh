#!/usr/bin/env bash
# Tune threshold τ via Optuna and evaluate ASK with Qwen3.5 models (0.8B and 2B).
# Output: results/ask_qwen35_{0.8b,2b}_results.json
#         results/ask_qwen35_{0.8b,2b}_episodes.csv
#         optuna.db
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_ask_qwen3] Tuning τ (15 Optuna trials) and evaluating ASK (Qwen3.5)"

python eval_ppo_slm.py --mode ask --slm qwen35-0.8b --wandb-group main
python eval_ppo_slm.py --mode ask --slm qwen35-2b   --wandb-group main

echo "[eval_ask_qwen3] Done → results/ask_qwen35_*.json"
