#!/usr/bin/env bash
# Runs all remaining evaluations in sequence.
# Safe to leave running unattended (tmux recommended).
#
# Order:
#   1. SLM-only  — Qwen3.5-0.8B, Qwen3.5-2B
#   2. ASK       — Qwen2.5-0.5B,  Qwen2.5-1.5B
#   3. ASK       — Qwen3.5-0.8B,  Qwen3.5-2B
#
# Total estimated runtime: ~12–20h
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "========================================"
echo " ASK-POMDP — Remaining Evaluations"
echo "========================================"

# --- 1. SLM-only: Qwen3.5 ---
echo ""
echo "[1/3] SLM-only — Qwen3.5-0.8B"
python eval_ppo_slm.py --mode slm --slm qwen35-0.8b --wandb-group main

echo ""
echo "[1/3] SLM-only — Qwen3.5-2B"
python eval_ppo_slm.py --mode slm --slm qwen35-2b --wandb-group main

# --- 2. ASK: Qwen2.5 ---
echo ""
echo "[2/3] ASK — Qwen2.5-0.5B (Optuna + eval)"
python eval_ppo_slm.py --mode ask --slm 0.5b --wandb-group main

echo ""
echo "[2/3] ASK — Qwen2.5-1.5B (Optuna + eval)"
python eval_ppo_slm.py --mode ask --slm 1.5b --wandb-group main

# --- 3. ASK: Qwen3.5 ---
echo ""
echo "[3/3] ASK — Qwen3.5-0.8B (Optuna + eval)"
python eval_ppo_slm.py --mode ask --slm qwen35-0.8b --wandb-group main

echo ""
echo "[3/3] ASK — Qwen3.5-2B (Optuna + eval)"
python eval_ppo_slm.py --mode ask --slm qwen35-2b --wandb-group main

echo ""
echo "========================================"
echo " All evaluations complete"
echo "========================================"
