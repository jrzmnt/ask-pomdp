#!/usr/bin/env bash
# Tune threshold τ via Optuna and evaluate ASK with Qwen3 models (0.8B and 2B).
# Output: results/ask_qwen3_{0.8b,2b}_results.json
#         results/ask_qwen3_{0.8b,2b}_episodes.csv
#         optuna.db
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_ask_qwen3] Tuning τ (15 Optuna trials) and evaluating ASK (Qwen3)"

python eval_ppo_slm.py --mode ask --slm qwen3-0.6b --wandb-group main
python eval_ppo_slm.py --mode ask --slm qwen3-1.7b   --wandb-group main

echo "[eval_ask_qwen3] Done → results/ask_qwen3_*.json"
