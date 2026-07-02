#!/usr/bin/env bash
# MC Dropout sample ablation for all three environments. Reuses each env's
# Optuna-selected τ (from {results,higher_lower/results,door_key/results}/thresholds.json).
# Produces tagged JSON+CSV files that plots/make_figures.py picks up for Fig 2.
#
# Parallelism: one job per GPU. Each job is a single
# (env, slm, N_MC) triple. A wait -n / GPU_PID bookkeeping loop keeps every
# GPU busy until the job list is drained.
#
# Usage:
#   bash scripts/ablation_mc_samples_all.sh                      # 8 GPUs, all envs+SLMs
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/ablation_mc_samples_all.sh
#   GPUS="0,1,2,3"          bash scripts/ablation_mc_samples_all.sh
#   ENVS="fourrooms doorkey"  SLMS="qwen3.5-2b" bash scripts/ablation_mc_samples_all.sh
#   N_LIST="5 10 20 30 50"  N_EP=50 bash scripts/ablation_mc_samples_all.sh
#   DOORKEY_SIZE=8         bash scripts/ablation_mc_samples_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-offline}"

# Defaults
ENVS="${ENVS:-fourrooms higherlower doorkey}"
SLMS="${SLMS:-qwen3.5-2b qwen3.5-4b}"
N_LIST="${N_LIST:-5 10 20 30 50}"
N_EP="${N_EP:-100}"
DOORKEY_SIZE="${DOORKEY_SIZE:-8}"
GROUP="${GROUP:-ablation_mc_samples}"

# Accept either space- or comma-separated GPU lists. Default: 8 GPUs.
read -ra GPUS <<< "$(echo "${GPUS:-0 1 2 3 4 5 6 7}" | tr ',' ' ')"

LOG_DIR="logs/ablation_mc_samples"
mkdir -p "$LOG_DIR"

# --- τ resolution -----------------------------------------------------------

best_tau () {
    local thr_file="$1" key="$2"
    python -c "
import json, sys
try:
    d = json.load(open('${thr_file}'))
except FileNotFoundError:
    print('')
    sys.exit(0)
e = d.get('${key}')
print(e['threshold'] if e else '')
"
}

slm_keytag () {
    case "$1" in
        qwen3.5-2b) echo "qwen35_2b" ;;
        qwen3.5-4b) echo "qwen35_4b" ;;
        random)     echo "random" ;;
        *)          echo "" ;;
    esac
}

env_thr_file () {
    case "$1" in
        fourrooms|fr)   echo "results/thresholds.json" ;;
        higherlower|hl) echo "higher_lower/results/thresholds.json" ;;
        doorkey|dk)     echo "door_key/results/thresholds.json" ;;
        *)              echo "" ;;
    esac
}

env_thr_key () {
    local env="$1" tag="$2"
    case "$env" in
        fourrooms|fr)   echo "fourrooms_${tag}" ;;
        higherlower|hl) echo "higherlower_${tag}" ;;
        doorkey|dk)     echo "doorkey_s${DOORKEY_SIZE}_${tag}" ;;
        *)              echo "" ;;
    esac
}

# --- job dispatch -----------------------------------------------------------

# Build the list of jobs up-front. Each job is "env|slm|tau|N".
JOBS=()
for env in $ENVS; do
    THR_FILE=$(env_thr_file "$env")
    if [ -z "$THR_FILE" ]; then
        echo "[warn] unknown env '$env'" >&2
        continue
    fi
    for SLM in $SLMS; do
        KEY_TAG=$(slm_keytag "$SLM")
        if [ -z "$KEY_TAG" ]; then
            echo "[err] unknown SLM '$SLM' (expected qwen3.5-2b / qwen3.5-4b / random)" >&2
            continue
        fi
        KEY=$(env_thr_key "$env" "$KEY_TAG")
        TAU=$(best_tau "$THR_FILE" "$KEY")
        if [ -z "$TAU" ]; then
            echo "[skip] ${env} ${SLM}: no τ in ${THR_FILE} (key=${KEY})" >&2
            continue
        fi
        for N in $N_LIST; do
            JOBS+=("${env}|${SLM}|${TAU}|${N}")
        done
    done
done

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "[err] no jobs to run" >&2
    exit 1
fi

echo "[ablation_mc] envs: $ENVS"
echo "[ablation_mc] SLMs: $SLMS"
echo "[ablation_mc] N values: $N_LIST  (per-env episodes: $N_EP)"
echo "[ablation_mc] GPUs: ${GPUS[*]}  ←  ${#JOBS[@]} jobs queued"
echo "[ablation_mc] logs: $LOG_DIR/"
echo ""

declare -A GPU_PID=()
declare -A PID_GPU=()
declare -A PID_LABEL=()
declare -A PID_LOG=()

dispatch_job () {
    local spec="$1" G="$2"
    IFS='|' read -r env SLM TAU N <<< "$spec"

    local TAG="mc${N}"
    local LABEL="${env}/${SLM}/N=${N}/τ=${TAU}"
    local LOG="${LOG_DIR}/$(echo "${env}_${SLM}_mc${N}" | tr '/.' '_')_gpu${G}.log"

    (
        set -euo pipefail
        echo "[gpu=$G] ===== ${LABEL} ====="
        case "$env" in
            fourrooms|fr)
                CUDA_VISIBLE_DEVICES="$G" python eval_ppo_slm.py \
                    --mode ask --slm "$SLM" \
                    --threshold "$TAU" --n-mc "$N" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP"
                ;;
            higherlower|hl)
                CUDA_VISIBLE_DEVICES="$G" python higher_lower/eval.py \
                    --mode ask --slm "$SLM" \
                    --threshold "$TAU" --n-mc "$N" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP"
                ;;
            doorkey|dk)
                CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
                    --mode ask --slm "$SLM" --size "$DOORKEY_SIZE" \
                    --threshold "$TAU" --n-mc "$N" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP"
                ;;
        esac
        echo "[gpu=$G] done ${LABEL}"
    ) >"$LOG" 2>&1 &

    local pid=$!
    GPU_PID[$G]="$pid"
    PID_GPU[$pid]="$G"
    PID_LABEL[$pid]="$LABEL"
    PID_LOG[$pid]="$LOG"
    echo "[dispatch] ${LABEL} → gpu=$G pid=$pid (log: $LOG)"
}

# Reap one finished child, free its GPU slot, and report status.
reap_one () {
    wait -n
    local rc=$?
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local g="${PID_GPU[$pid]}"
            local lbl="${PID_LABEL[$pid]}"
            local log="${PID_LOG[$pid]:-}"
            unset "GPU_PID[$g]"
            unset "PID_GPU[$pid]"
            unset "PID_LABEL[$pid]"
            unset "PID_LOG[$pid]"
            if [ "$rc" -eq 0 ]; then
                echo "[done] pid=$pid ${lbl} on gpu=$g (rc=0)"
            else
                echo "[fail] pid=$pid ${lbl} on gpu=$g (rc=$rc) → ${log}" >&2
                fail=1
            fi
            return 0
        fi
    done
    return 0
}

fail=0

for spec in "${JOBS[@]}"; do
    free_gpu=""
    while [ -z "$free_gpu" ]; do
        for g in "${GPUS[@]}"; do
            if [ -z "${GPU_PID[$g]:-}" ]; then
                free_gpu="$g"
                break
            fi
        done
        if [ -z "$free_gpu" ]; then
            reap_one
        fi
    done
    dispatch_job "$spec" "$free_gpu"
done

echo ""
echo "[ablation_mc] all jobs dispatched; waiting for completion."
echo "  follow logs with:  tail -F ${LOG_DIR}/*.log"
echo ""

while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

if [ "$fail" -ne 0 ]; then
    echo "[ablation_mc] FINISHED WITH ERRORS — inspect ${LOG_DIR}/*.log" >&2
    exit 1
fi

echo ""
echo "[ablation_mc] done. Replot Fig 2 with:  python plots/make_figures.py fig2"
