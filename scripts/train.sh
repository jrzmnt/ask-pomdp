#!/usr/bin/env bash
# Train PPO with MC Dropout on MiniGrid-FourRooms-v0.
# Output: runs/ppo/model.zip  and  runs/ppo/best_model/best_model.zip
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[train] Starting PPO training on MiniGrid-FourRooms-v0"
echo "[train] Config: configs/rl/ppo.yaml"
echo ""

python train_ppo.py

echo ""
echo "[train] Done. Model saved to runs/ppo/model.zip"
