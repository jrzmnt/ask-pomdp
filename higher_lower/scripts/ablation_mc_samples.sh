#!/usr/bin/env bash
# Ablation: MC Dropout forward passes (N) for HigherLower — Qwen3.5-2B and Qwen3.5-4B.
# Reads best τ from higher_lower/results/thresholds.json (key: higherlower_{model_tag}).
# Prerequisite: run higher_lower/scripts/eval_ask.sh first.
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

for SLM in qwen3.5-2b qwen3.5-4b; do
    if [ "${SLM}" = "qwen3.5-2b" ]; then
        KEY="higherlower_qwen35_2b"
    else
        KEY="higherlower_qwen35_4b"
    fi

    BEST_TAU=$(python -c "
import json
d = json.load(open('${THRESHOLD_FILE}'))
print(d['${KEY}']['threshold'])
")
    echo ""
    echo "=== ${SLM}  τ=${BEST_TAU} (key: ${KEY}) ==="

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
