#!/usr/bin/env bash
# Evaluate SLM-only baseline with Qwen3.5 models (0.8B and 2B).
# Thinking mode is off by default for these sizes — no extra config needed.
# Output: results/slm_qwen35_{0.8b,2b}_results.json
#         results/slm_qwen35_{0.8b,2b}_episodes.csv
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_slm_qwen3] Evaluating SLM-only (Qwen3.5-0.8B and Qwen3.5-2B)"

python eval_ppo_slm.py --mode slm --slm qwen35-0.8b --wandb-group main
python eval_ppo_slm.py --mode slm --slm qwen35-2b   --wandb-group main

echo "[eval_slm_qwen3] Done → results/slm_qwen35_*.json"
