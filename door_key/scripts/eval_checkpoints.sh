#!/usr/bin/env bash
# Ablation: PPO optimality effect on ASK — DoorKey.
# Evaluates PPO + ASK (Qwen3.5-2B and Qwen3.5-4B, stateful + rationale prompts)
# for each reward-threshold checkpoint.
#
# Parallelism: one checkpoint per GPU (default 4 GPUs → 4 checkpoints in flight).
# Each subshell runs its PPO eval and then the two ASK evals sequentially, so the
# Optuna search for the 2B and 4B SLMs never share a GPU.
#
# Checkpoints saved by RewardThresholdCheckpointCallback in door_key/train.py:
#   model_reward_030.zip  (reward ≥ 0.30)
#   model_reward_050.zip  (reward ≥ 0.50)
#   model_reward_070.zip  (reward ≥ 0.70)
#   model_reward_090.zip  (reward ≥ 0.90)
#
# Usage:
#   bash door_key/scripts/eval_checkpoints.sh                  # size=5, GPUs=0,1,2,3
#   SIZE=8 bash door_key/scripts/eval_checkpoints.sh
#   GPUS="0 1 2 3" SIZE=5 bash door_key/scripts/eval_checkpoints.sh
#   SLMS="qwen3.5-2b" bash door_key/scripts/eval_checkpoints.sh
#
# Output: door_key/results/  +  logs/door_key_checkpoints/ckpt_rNNN_gpuG.log
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SIZE="${SIZE:-5}"
# Accept either space- or comma-separated lists for these env vars.
read -ra REWARDS <<< "$(echo "${REWARDS:-030 050 070 090}" | tr ',' ' ')"
read -ra GPUS    <<< "$(echo "${GPUS:-0 1 2 3}"            | tr ',' ' ')"
read -ra SLMS    <<< "$(echo "${SLMS:-qwen3.5-2b qwen3.5-4b}" | tr ',' ' ')"
N_OPTUNA_TRIALS="${N_OPTUNA_TRIALS:-10}"
GROUP="${GROUP:-doorkey_checkpoints}"
PROMPT_STYLE="${PROMPT_STYLE:-stateful}"
PROMPT_RATIONALE="${PROMPT_RATIONALE:-1}"

CKPT_DIR="runs/door_key/checkpoints_s${SIZE}"
LOG_DIR="logs/door_key_checkpoints"
mkdir -p "$LOG_DIR"

if [ "${#REWARDS[@]}" -gt "${#GPUS[@]}" ]; then
    echo "[warn] ${#REWARDS[@]} checkpoints but only ${#GPUS[@]} GPUs;" \
         "extra checkpoints will wait for the first free slot." >&2
fi

extra_args=(--prompt-style "$PROMPT_STYLE")
if [ "$PROMPT_RATIONALE" = "1" ] || [ "$PROMPT_RATIONALE" = "true" ]; then
    extra_args+=(--prompt-rationale)
fi

echo "[eval_checkpoints] DoorKey size=${SIZE}"
echo "  Checkpoints: ${REWARDS[*]}"
echo "  GPUs:        ${GPUS[*]}"
echo "  SLMs:        ${SLMS[*]}"
echo "  Prompt:      ${extra_args[*]}"
echo "  Logs:        ${LOG_DIR}/"
echo ""

declare -A GPU_PID=()    # gpu_id  → pid currently using it
declare -A PID_GPU=()    # pid     → gpu_id
declare -A PID_LABEL=()  # pid     → human label
declare -A PID_LOG=()    # pid     → log file

dispatch () {
    local R="$1" G="$2"
    local MODEL="${CKPT_DIR}/model_reward_${R}"
    local TAG="ckpt_r${R}"
    local LABEL="0.${R}"
    local LOG="${LOG_DIR}/ckpt_r${R}_s${SIZE}_gpu${G}.log"

    if [ ! -f "${MODEL}.zip" ]; then
        echo "[skip] ${MODEL}.zip not found"
        return 1
    fi

    (
        set -euo pipefail
        echo "[gpu=$G] ===== checkpoint reward ≥ ${LABEL} ====="

        echo "[gpu=$G] PPO eval"
        CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
            --mode ppo --size "$SIZE" \
            --model-path "$MODEL" --tag "$TAG" \
            --wandb-group "$GROUP"

        for SLM in "${SLMS[@]}"; do
            echo "[gpu=$G] ASK eval (${SLM})"
            CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
                --mode ask --size "$SIZE" \
                --slm "$SLM" \
                --model-path "$MODEL" --tag "$TAG" \
                --n-optuna-trials "$N_OPTUNA_TRIALS" \
                --wandb-group "$GROUP" \
                "${extra_args[@]}"
        done

        echo "[gpu=$G] done ckpt ${LABEL}"
    ) >"$LOG" 2>&1 &

    local pid=$!
    GPU_PID[$G]="$pid"
    PID_GPU[$pid]="$G"
    PID_LABEL[$pid]="ckpt=$LABEL"
    PID_LOG[$pid]="$LOG"
    echo "[dispatch] ckpt ${LABEL} → gpu=$G pid=$pid (log: $LOG)"
}

# Reap one finished child, free its GPU slot, and report status.
reap_one () {
    wait -n
    local rc=$?
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local g="${PID_GPU[$pid]}"
            local lbl="${PID_LABEL[$pid]}"
            unset "GPU_PID[$g]"
            unset "PID_GPU[$pid]"
            unset "PID_LABEL[$pid]"
            unset "PID_LOG[$pid]"
            if [ "$rc" -eq 0 ]; then
                echo "[done] pid=$pid $lbl on gpu=$g (rc=0)"
            else
                echo "[fail] pid=$pid $lbl on gpu=$g (rc=$rc) → ${PID_LOG[$pid]:-}" >&2
                fail=1
            fi
            return 0
        fi
    done
    return 0
}

fail=0

for R in "${REWARDS[@]}"; do
    # Wait until a GPU slot is free.
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
    dispatch "$R" "$free_gpu" || true
done

echo ""
echo "[eval_checkpoints] all jobs dispatched; waiting for completion."
echo "  follow logs with:  tail -F ${LOG_DIR}/*.log"
echo ""

while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

if [ "$fail" -ne 0 ]; then
    echo "[eval_checkpoints] FINISHED WITH ERRORS — inspect ${LOG_DIR}/*.log" >&2
    exit 1
fi

echo "[eval_checkpoints] Done → door_key/results/  (logs: ${LOG_DIR}/)"
