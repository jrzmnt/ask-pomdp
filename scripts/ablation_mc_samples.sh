#!/usr/bin/env bash
# Ablation: number of MC Dropout forward passes (N).
#
# Uses the best τ found by Optuna (reads from results/ask_qwen_1.5b_results.json).
# Sweeps N ∈ {5, 10, 20, 30, 50}.
# Shows the trade-off between uncertainty quality and inference cost.
#
# Prerequisite: run eval_ask.sh first to obtain the best threshold.
#
# Output: results/ask_qwen_1.5b_results_mc{N}.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

RESULT_FILE="results/ask_qwen_1.5b_results.json"

if [ ! -f "${RESULT_FILE}" ]; then
    echo "[ablation_mc_samples] ERROR: ${RESULT_FILE} not found."
    echo "  Run scripts/eval_ask.sh first to obtain the best threshold."
    exit 1
fi

BEST_TAU=$(python -c "import json; d=json.load(open('${RESULT_FILE}')); print(d['threshold'])")
echo "[ablation_mc_samples] Best τ = ${BEST_TAU} (from ${RESULT_FILE})"
echo "[ablation_mc_samples] Sweeping N MC samples for Qwen 1.5B"

for N in 5 10 20 30 50; do
    echo ""
    echo "  N = ${N}"
    python eval_ppo_slm.py \
        --mode ask \
        --slm 1.5b \
        --threshold "${BEST_TAU}" \
        --n-mc "${N}" \
        --tag "mc${N}" \
        --wandb-group ablation_mc_samples
done

echo ""
echo "[ablation_mc_samples] Done → results/ask_qwen_1.5b_results_mc*.json"
