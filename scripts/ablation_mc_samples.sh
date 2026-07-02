#!/usr/bin/env bash
# Ablation: MC Dropout forward passes (N) — Qwen3.5-2B and Qwen3.5-4B.
# Reads best τ from results/thresholds.json (key: fourrooms_{model_tag}).
# Prerequisite: run scripts/eval_ask.sh first.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

THRESHOLD_FILE="results/thresholds.json"

if [ ! -f "${THRESHOLD_FILE}" ]; then
    echo "[ablation_mc_samples] ERROR: ${THRESHOLD_FILE} not found."
    echo "  Run scripts/eval_ask.sh first."
    exit 1
fi

for SLM in qwen3.5-2b qwen3.5-4b; do
    if [ "${SLM}" = "qwen3.5-2b" ]; then
        KEY="fourrooms_qwen35_2b"
    else
        KEY="fourrooms_qwen35_4b"
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
        python eval_ppo_slm.py \
            --mode ask \
            --slm "${SLM}" \
            --threshold "${BEST_TAU}" \
            --n-mc "${N}" \
            --tag "mc${N}" \
            --wandb-group ablation_mc_samples
    done
done

echo ""
echo "[ablation_mc_samples] Done → results/ask_*_results_mc*.json"
