#!/usr/bin/env bash
# Ablation: number of MC Dropout forward passes (N).
#
# Uses the best τ found by Optuna (reads from results/thresholds.json).
# Sweeps N ∈ {5, 10, 20, 30, 50} for one representative per model family:
#   - Qwen2.5-1.5B
#   - Qwen3.5-2B
#
# Prerequisite: run eval_ask.sh and eval_ask_qwen3.sh first.
#
# Output: results/ask_*_results_mc{N}.json
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

THRESHOLD_FILE="results/thresholds.json"

if [ ! -f "${THRESHOLD_FILE}" ]; then
    echo "[ablation_mc_samples] ERROR: ${THRESHOLD_FILE} not found."
    echo "  Run eval_ask.sh and eval_ask_qwen3.sh first."
    exit 1
fi

for SLM in 1.5b qwen35-2b; do
    # Resolve HF model name → look up threshold
    if [ "${SLM}" = "1.5b" ]; then
        MODEL_KEY="Qwen/Qwen2.5-1.5B-Instruct"
    else
        MODEL_KEY="Qwen/Qwen3.5-2B"
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
