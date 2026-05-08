#!/usr/bin/env bash
# Smoke test: verifies the full pipeline runs end-to-end without errors.
#
# Stages:
#   1. pytest unit/integration tests (mock SLM, no GPU required)
#   2. Train PPO for 2 000 steps (CPU, ~30s)
#   3. Eval PPO for 3 episodes
#
# W&B is set to offline mode — no upload happens.
# Runtime: ~2-3 minutes.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export WANDB_MODE=offline

echo "========================================"
echo " ASK-POMDP Smoke Test"
echo "========================================"

# --- 1. pytest ---
echo ""
echo "[1/3] Running unit tests..."
python -m pytest tests/smoke_test.py -v --tb=short
echo "      Tests passed."

# --- 2. Train PPO (tiny config) ---
echo ""
echo "[2/3] Training PPO (2 000 steps, CPU)..."
python train_ppo.py --config configs/rl/ppo_smoke.yaml
echo "      Training passed."

# --- 3. Eval PPO ---
echo ""
echo "[3/3] Evaluating PPO (3 episodes)..."
python eval_ppo_slm.py --mode ppo --n-episodes 3 --wandb-group smoke
echo "      Eval passed."

echo ""
echo "========================================"
echo " Smoke test complete — all checks passed"
echo "========================================"
