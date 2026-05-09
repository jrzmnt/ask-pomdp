#!/usr/bin/env bash
# Evaluate SLM-only baseline with Qwen3 models (0.8B and 2B).
# Thinking mode is off by default for these sizes — no extra config needed.
# Output: results/slm_qwen3_{0.8b,2b}_results.json
#         results/slm_qwen3_{0.8b,2b}_episodes.csv
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[eval_slm_qwen3] Evaluating SLM-only (Qwen3-0.8B and Qwen3-2B)"

python eval_ppo_slm.py --mode slm --slm qwen3-0.6b --wandb-group main
python eval_ppo_slm.py --mode slm --slm qwen3-1.7b   --wandb-group main

echo "[eval_slm_qwen3] Done → results/slm_qwen3_*.json"
