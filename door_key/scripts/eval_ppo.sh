#!/usr/bin/env bash
# Evaluate PPO baseline on DoorKey.
# Example: bash door_key/scripts/eval_ppo.sh --size 5
python door_key/eval.py --mode ppo "$@"
