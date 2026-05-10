#!/usr/bin/env bash
# Ablation: number of MC Dropout forward passes (N) for HigherLower.
#
# Uses the best τ found by Optuna (reads from higher_lower/results/thresholds.json).
# Sweeps N ∈ {5, 10, 20, 30, 50} for one representative per model family:
#   - Qwen2.5-1.5B
#   - Qwen3-1.7B
#
# Prerequisite: run eval_ask.sh first.
#
# Output: higher_lower/results/ask_*_results_mc{N}.json
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

THRESHOLD_FILE="higher_lower/results/thresholds.json"

if [ ! -f "${THRESHOLD_FILE}" ]; then
    echo "[hl_ablation_mc_samples] ERROR: ${THRESHOLD_FILE} not found."
    echo "  Run higher_lower/scripts/eval_ask.sh first."
    exit 1
fi

for SLM in 1.5b qwen3-1.7b; do
    if [ "${SLM}" = "1.5b" ]; then
        MODEL_KEY="Qwen/Qwen2.5-1.5B-Instruct"
    else
        MODEL_KEY="Qwen/Qwen3-1.7B"
    fi

    BEST_TAU=$(python -c "
import json
d = json.load(open('${THRESHOLD_FILE}'))
print(d['${MODEL_KEY}']['threshold'])
")
    echo ""
    echo "=== Model: ${SLM}  τ=${BEST_TAU} ==="

    for N in 5 10 20 30 50; do
        echo "  N = ${N}"
        python higher_lower/eval.py \
            --mode ask \
            --slm "${SLM}" \
            --threshold "${BEST_TAU}" \
            --n-mc "${N}" \
            --tag "mc${N}" \
            --wandb-group hl_ablation_mc_samples
    done
done

echo ""
echo "[hl_ablation_mc_samples] Done → higher_lower/results/ask_*_results_mc*.json"
