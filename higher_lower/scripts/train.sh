#!/usr/bin/env bash
# Train PPO on HigherLower (~5–10 min)
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
echo "[hl_train] Training PPO on HigherLower..."
python higher_lower/train.py --timesteps 500000 --seed 42
echo "[hl_train] Done → runs/higher_lower/model.zip"
