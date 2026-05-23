#!/usr/bin/env bash
# Finish the remaining PPO-optimality cells in the README.
#
# As of this commit, the only experiments still missing for the ablation tables
# are:
#   FourRooms — ckpt r070 × {Qwen3.5-2B, Qwen3.5-4B}             (writes results/ask_qwen35_{2b,4b}_results_ckpt_r070.json)
#   DoorKey-8 — ckpt {r030, r050, r070} × Qwen3.5-4B             (writes door_key/results/ask_qwen35_4b_results_s8_ckpt_r{030,050,070}.json)
#
# 5 jobs total → dispatched in parallel across 8 GPUs (one job per GPU).
# Same wait -n / GPU_PID bookkeeping pattern used in scripts/ablation_mc_samples_all.sh
# and door_key/scripts/eval_checkpoints.sh.
#
# Each ASK job runs Optuna with N_OPTUNA_TRIALS trials (10 by default), reuses the
# stateful+rationale prompt, and produces a tagged JSON+CSV next to the existing
# checkpoint results.
#
# Usage:
#   bash scripts/finish_pending_ablations.sh                 # 8 GPUs (default)
#   GPUS="0 1 2 3 4 5 6 7"  bash scripts/finish_pending_ablations.sh
#   GPUS="0,1,2,3"          bash scripts/finish_pending_ablations.sh
#   N_OPTUNA_TRIALS=20      bash scripts/finish_pending_ablations.sh
#   N_EP=50                 bash scripts/finish_pending_ablations.sh    # validation episodes per Optuna trial

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-offline}"

read -ra GPUS <<< "$(echo "${GPUS:-0 1 2 3 4 5 6 7}" | tr ',' ' ')"

N_OPTUNA_TRIALS="${N_OPTUNA_TRIALS:-10}"
N_EP="${N_EP:-100}"
DOORKEY_SIZE="${DOORKEY_SIZE:-8}"

FR_CKPT_DIR="runs/ppo/checkpoints"
DK_CKPT_DIR="runs/door_key/checkpoints_s${DOORKEY_SIZE}"

FR_GROUP="${FR_GROUP:-fourrooms_checkpoints}"
DK_GROUP="${DK_GROUP:-doorkey_checkpoints}"

LOG_DIR="logs/finish_pending"
mkdir -p "$LOG_DIR"

# --- job list ---------------------------------------------------------------
# Each entry: "env|slm|ckpt_label|model_path"
JOBS=(
    "fourrooms|qwen3.5-2b|r070|${FR_CKPT_DIR}/model_reward_070"
    "fourrooms|qwen3.5-4b|r070|${FR_CKPT_DIR}/model_reward_070"
    "doorkey|qwen3.5-4b|r030|${DK_CKPT_DIR}/model_reward_030"
    "doorkey|qwen3.5-4b|r050|${DK_CKPT_DIR}/model_reward_050"
    "doorkey|qwen3.5-4b|r070|${DK_CKPT_DIR}/model_reward_070"
)

# Filter out missing checkpoint files up front so we don't dispatch dead jobs.
ALIVE=()
for spec in "${JOBS[@]}"; do
    IFS='|' read -r env slm ckpt model <<< "$spec"
    if [ ! -f "${model}.zip" ]; then
        echo "[skip] ${env}/${slm}/${ckpt} → ${model}.zip not found" >&2
        continue
    fi
    ALIVE+=("$spec")
done
JOBS=("${ALIVE[@]}")

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "[err] no jobs to run; all checkpoint files missing" >&2
    exit 1
fi

echo "[finish_pending] GPUs:   ${GPUS[*]}"
echo "[finish_pending] jobs:   ${#JOBS[@]}"
echo "[finish_pending] Optuna: ${N_OPTUNA_TRIALS} trials × ${N_EP} ep / trial"
echo "[finish_pending] logs:   ${LOG_DIR}/"
echo ""

declare -A GPU_PID=()
declare -A PID_GPU=()
declare -A PID_LABEL=()
declare -A PID_LOG=()

dispatch_job () {
    local spec="$1" G="$2"
    IFS='|' read -r env slm ckpt model <<< "$spec"
    local TAG="ckpt_${ckpt}"
    local LABEL="${env}/${slm}/${ckpt}"
    local LOG="${LOG_DIR}/$(echo "${env}_${slm}_${ckpt}" | tr '/.' '_')_gpu${G}.log"

    (
        set -euo pipefail
        echo "[gpu=$G] ===== ${LABEL} ====="
        case "$env" in
            fourrooms)
                CUDA_VISIBLE_DEVICES="$G" python eval_ppo_slm.py \
                    --mode ask --slm "$slm" \
                    --model-path "$model" --tag "$TAG" \
                    --n-optuna-trials "$N_OPTUNA_TRIALS" \
                    --n-episodes "$N_EP" \
                    --prompt-style stateful --prompt-rationale \
                    --wandb-group "$FR_GROUP"
                ;;
            doorkey)
                CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
                    --mode ask --slm "$slm" --size "$DOORKEY_SIZE" \
                    --model-path "$model" --tag "$TAG" \
                    --n-optuna-trials "$N_OPTUNA_TRIALS" \
                    --n-episodes "$N_EP" \
                    --prompt-style stateful --prompt-rationale \
                    --wandb-group "$DK_GROUP"
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
echo "[finish_pending] all jobs dispatched; waiting for completion."
echo "  follow logs with:  tail -F ${LOG_DIR}/*.log"
echo ""

while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

if [ "$fail" -ne 0 ]; then
    echo "[finish_pending] FINISHED WITH ERRORS — inspect ${LOG_DIR}/*.log" >&2
    exit 1
fi

echo ""
echo "[finish_pending] done. New JSON/CSV files:"
echo "  results/ask_qwen35_{2b,4b}_results_ckpt_r070.json"
echo "  door_key/results/ask_qwen35_4b_results_s8_ckpt_r{030,050,070}.json"
echo "Update the README PPO optimality tables next."
