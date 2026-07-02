#!/usr/bin/env bash
# Dense τ sweep for Fig 1 across all three environments. Runs ASK at a grid of
# fixed τ values (no Optuna) on the FULL PPO model so reward and intervention
# rate are measured under a single, fixed policy. Produces tagged JSON+CSV
# files that plots/make_figures.py picks up:
#
#   results/ask_{tag}_results_threshold_{TAU/./}.json
#   higher_lower/results/ask_{tag}_results_threshold_{TAU/./}.json
#   door_key/results/ask_{tag}_results_s{SIZE}_threshold_{TAU/./}.json
#
# Parallelism: one job per GPU. Each job is a single (env, slm, τ) triple.
# A wait -n / GPU_PID bookkeeping loop keeps every GPU busy until the
# job list is drained.
#
# Usage:
#   bash scripts/ablation_threshold_all.sh                                # 8 GPUs default
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/ablation_threshold_all.sh
#   GPUS="0,1,2,3"          bash scripts/ablation_threshold_all.sh
#   ENVS="fourrooms doorkey"  SLMS="qwen3.5-2b" bash scripts/ablation_threshold_all.sh
#   FR_TAUS="0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4 1.6 1.8 2.0" N_EP=50 bash ...
#   DK_TAUS="0.5 1.0 1.2 1.4 1.6 1.8 2.0" DOORKEY_SIZE=8 bash ...
#
# With the default config the job count is:
#   FR: 11 τ × 2 SLMs = 22 jobs
#   HL: 10 τ × 2 SLMs = 20 jobs
#   DK:  8 τ × 2 SLMs = 16 jobs
#   ⇒ 58 jobs across 8 GPUs (~7-8 waves).

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-offline}"

# Defaults. τ grids are informed by the current Optuna best-τ per env.
ENVS="${ENVS:-fourrooms higherlower doorkey}"
SLMS="${SLMS:-qwen3.5-2b qwen3.5-4b}"
N_EP="${N_EP:-100}"
DOORKEY_SIZE="${DOORKEY_SIZE:-8}"
GROUP="${GROUP:-ablation_threshold}"

FR_TAUS="${FR_TAUS:-0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.3 1.7}"   # full PPO τ* ≈ 0.46–0.48
HL_TAUS="${HL_TAUS:-0.005 0.02 0.05 0.1 0.2 0.4 0.6 0.9 1.2 1.5}"     # full PPO τ* ≈ 0.013
DK_TAUS="${DK_TAUS:-0.3 0.6 0.9 1.1 1.3 1.5 1.7 1.9}"                  # full PPO τ* ≈ 1.45–1.53

# Accept either space- or comma-separated GPU lists. Default: 8 GPUs.
read -ra GPUS <<< "$(echo "${GPUS:-0 1 2 3 4 5 6 7}" | tr ',' ' ')"

LOG_DIR="logs/ablation_threshold"
mkdir -p "$LOG_DIR"

# --- build job list ---------------------------------------------------------

# Each job spec: "env|slm|tau"
JOBS=()
for SLM in $SLMS; do
    for env in $ENVS; do
        case "$env" in
            fourrooms|fr)
                for TAU in $FR_TAUS; do JOBS+=("fourrooms|${SLM}|${TAU}"); done ;;
            higherlower|hl)
                for TAU in $HL_TAUS; do JOBS+=("higherlower|${SLM}|${TAU}"); done ;;
            doorkey|dk)
                for TAU in $DK_TAUS; do JOBS+=("doorkey|${SLM}|${TAU}"); done ;;
            *) echo "[warn] unknown env '$env'" >&2 ;;
        esac
    done
done

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "[err] no jobs to run" >&2
    exit 1
fi

echo "[ablation_threshold] envs: $ENVS"
echo "[ablation_threshold] SLMs: $SLMS"
echo "[ablation_threshold] FR τ: $FR_TAUS"
echo "[ablation_threshold] HL τ: $HL_TAUS"
echo "[ablation_threshold] DK τ: $DK_TAUS  (size=${DOORKEY_SIZE})"
echo "[ablation_threshold] episodes per τ: $N_EP"
echo "[ablation_threshold] GPUs: ${GPUS[*]}  ←  ${#JOBS[@]} jobs queued"
echo "[ablation_threshold] logs: $LOG_DIR/"
echo ""

declare -A GPU_PID=()
declare -A PID_GPU=()
declare -A PID_LABEL=()
declare -A PID_LOG=()

dispatch_job () {
    local spec="$1" G="$2"
    IFS='|' read -r env SLM TAU <<< "$spec"

    local TAG="threshold_${TAU/./}"
    local LABEL="${env}/${SLM}/τ=${TAU}"
    local LOG="${LOG_DIR}/$(echo "${env}_${SLM}_${TAG}" | tr '/.' '_')_gpu${G}.log"

    (
        set -euo pipefail
        echo "[gpu=$G] ===== ${LABEL} ====="
        case "$env" in
            fourrooms|fr)
                CUDA_VISIBLE_DEVICES="$G" python eval_ppo_slm.py \
                    --mode ask --slm "$SLM" \
                    --threshold "$TAU" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP" --prompt-style stateful --prompt-rationale
                ;;
            higherlower|hl)
                CUDA_VISIBLE_DEVICES="$G" python higher_lower/eval.py \
                    --mode ask --slm "$SLM" \
                    --threshold "$TAU" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP" --prompt-style stateful --prompt-rationale
                ;;
            doorkey|dk)
                CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
                    --mode ask --slm "$SLM" --size "$DOORKEY_SIZE" \
                    --threshold "$TAU" --n-episodes "$N_EP" \
                    --tag "$TAG" --wandb-group "$GROUP" --prompt-style stateful --prompt-rationale
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
echo "[ablation_threshold] all jobs dispatched; waiting for completion."
echo "  follow logs with:  tail -F ${LOG_DIR}/*.log"
echo ""

while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

if [ "$fail" -ne 0 ]; then
    echo "[ablation_threshold] FINISHED WITH ERRORS — inspect ${LOG_DIR}/*.log" >&2
    exit 1
fi

echo ""
echo "[ablation_threshold] done. Replot Fig 1 with:  python plots/make_figures.py fig1"
