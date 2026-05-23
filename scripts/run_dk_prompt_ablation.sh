#!/usr/bin/env bash
# DoorKey SLM-only prompt ablation, parallel over 8 GPUs.
#
# Sweeps prompt style ∈ {basic, enriched, stateful_min, stateful, stateful+rationale}
# for both Qwen3.5-2B and Qwen3.5-4B (override SLMS=... to subset).
# Defaults: 100 episodes per run, 5 styles × 2 SLMs = 10 jobs (~2 waves on 8 GPUs).
#
# Same wait -n / GPU_PID scheduler as the other parallel ablation scripts in
# this repo (scripts/ablation_mc_samples_all.sh, scripts/ablation_threshold_all.sh,
# scripts/ablation_prompt_ask_all.sh, door_key/scripts/eval_checkpoints.sh).
#
# Usage:
#   bash scripts/run_dk_prompt_ablation.sh                                # size=5, 8 GPUs, both SLMs
#   SIZE=8 bash scripts/run_dk_prompt_ablation.sh
#   GPUS="0 1 2 3 4 5 6 7" SIZE=8 bash scripts/run_dk_prompt_ablation.sh
#   GPUS="0,1,2,3"          SIZE=8 bash scripts/run_dk_prompt_ablation.sh
#   SLMS="qwen3.5-2b"  N_EP=25 bash scripts/run_dk_prompt_ablation.sh     # smaller smoke run
#
# Output:
#   door_key/results/slm_*_results_s${SIZE}_<tag>.json
#   door_key/results/slm_*_episodes_s${SIZE}_<tag>.csv
#   logs/run_dk_prompt_ablation/<slm>_<tag>_s${SIZE}_gpu<G>.log

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$(pwd)"

SLMS="${SLMS:-qwen3.5-2b qwen3.5-4b}"
SIZE="${SIZE:-5}"
N_EP="${N_EP:-100}"
GROUP="${GROUP:-doorkey_prompt_ablation}"

# Accept either space- or comma-separated GPU lists. Default: 8 GPUs.
read -ra GPUS <<< "$(echo "${GPUS:-0 1 2 3 4 5 6 7}" | tr ',' ' ')"

# Prompt configs: TAG|STYLE|RATIONALE(0/1)
PROMPTS=(
    "basic|basic|0"
    "enriched|enriched|0"
    "stateful_min|stateful_min|0"
    "stateful|stateful|0"
    "stateful_rat|stateful|1"
)

LOG_DIR="logs/run_dk_prompt_ablation"
mkdir -p "$LOG_DIR"

# Map CLI SLM name → results filename short-name. Mirrors short_model_name()
# in door_key/eval.py.
slm_short () {
    case "$1" in
        qwen3.5-2b) echo "qwen35_2b" ;;
        qwen3.5-4b) echo "qwen35_4b" ;;
        random)     echo "random" ;;
        *)          echo "$1" | tr -c 'a-zA-Z0-9' '_' ;;
    esac
}

# Returns 0 if the (slm, tag, style, rationale) cell is already covered by an
# existing JSON file with at least N_EP episodes. The un-tagged default JSON
# (door_key/results/slm_<slm>_results_s<SIZE>.json) is assumed to be the
# `basic` prompt without rationale (per the README and door_key/eval.py
# default CLI), unless it carries explicit prompt_style metadata.
already_done_dk () {
    local SLM_SHORT="$1" TAG="$2" STYLE="$3" RAT="$4"
    local TAGGED="door_key/results/slm_${SLM_SHORT}_results_s${SIZE}_${TAG}.json"
    local UNTAGGED="door_key/results/slm_${SLM_SHORT}_results_s${SIZE}.json"
    python - "$TAGGED" "$UNTAGGED" "$STYLE" "$RAT" "$N_EP" <<'PY'
import json, sys, os
tagged, untagged, style, rat, n_ep = sys.argv[1:6]
n_ep = int(n_ep)
want_rat = rat in ("1", "true", "True")
def n_ok(d): return d.get("n_episodes", 0) >= n_ep
def load(p):
    try: return json.load(open(p))
    except Exception: return None
d = load(tagged)
if d and n_ok(d): sys.exit(0)
d = load(untagged)
if d and n_ok(d):
    got_style = d.get("prompt_style", "basic")            # default = basic
    got_rat   = bool(d.get("prompt_rationale", False))
    if got_style == style and got_rat == want_rat:
        sys.exit(0)
sys.exit(1)
PY
}

# --- build job list (filtering out already-done cells) ---------------------
JOBS=()
SKIPPED=()
for SLM in $SLMS; do
    SLM_SHORT=$(slm_short "$SLM")
    for P in "${PROMPTS[@]}"; do
        IFS='|' read -r TAG STYLE RAT <<< "$P"
        if already_done_dk "$SLM_SHORT" "$TAG" "$STYLE" "$RAT"; then
            SKIPPED+=("${SLM}/${TAG}")
            continue
        fi
        JOBS+=("${SLM}|${TAG}|${STYLE}|${RAT}")
    done
done

if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "[dk_prompt] skipping ${#SKIPPED[@]} cell(s) already covered by existing JSONs:"
    for s in "${SKIPPED[@]}"; do echo "    skip ${s}"; done
fi

if [ "${#JOBS[@]}" -eq 0 ]; then
    echo "[dk_prompt] nothing to do — all requested cells already present." >&2
    exit 0
fi

echo "[dk_prompt] SLMs: $SLMS"
echo "[dk_prompt] prompts: ${PROMPTS[*]}  (episodes per run: $N_EP, size: $SIZE)"
echo "[dk_prompt] GPUs: ${GPUS[*]}  ←  ${#JOBS[@]} jobs queued"
echo "[dk_prompt] logs: $LOG_DIR/"
echo ""

declare -A GPU_PID=()
declare -A PID_GPU=()
declare -A PID_LABEL=()
declare -A PID_LOG=()

dispatch_job () {
    local spec="$1" G="$2"
    IFS='|' read -r SLM TAG STYLE RAT <<< "$spec"
    local LABEL="${SLM}/${TAG}/s${SIZE}"
    local LOG="${LOG_DIR}/$(echo "${SLM}_${TAG}_s${SIZE}" | tr '/.' '_')_gpu${G}.log"

    local extra=(--prompt-style "$STYLE")
    if [ "$RAT" = "1" ] || [ "$RAT" = "true" ]; then
        extra+=(--prompt-rationale)
    fi

    (
        set -euo pipefail
        echo "[gpu=$G] ===== ${LABEL} ====="
        CUDA_VISIBLE_DEVICES="$G" python door_key/eval.py \
            --mode slm --slm "$SLM" --size "$SIZE" \
            --n-episodes "$N_EP" \
            --tag "$TAG" --wandb-group "$GROUP" \
            "${extra[@]}"
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
echo "[dk_prompt] all jobs dispatched; waiting for completion."
echo "  follow logs with:  tail -F ${LOG_DIR}/*.log"
echo ""

while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

if [ "$fail" -ne 0 ]; then
    echo "[dk_prompt] FINISHED WITH ERRORS — inspect ${LOG_DIR}/*.log" >&2
    exit 1
fi

echo ""
echo "[dk_prompt] done. Results under door_key/results/slm_*_s${SIZE}_<tag>.{json,csv}  (W&B group $GROUP)"
