#!/usr/bin/env bash
# Evaluate ASK (PPO + SLM gated by MC Dropout) on DoorKey.
# Runs Optuna to find optimal tau if --threshold is not provided.
# Example: bash door_key/scripts/eval_ask.sh --size 5
python door_key/eval.py --mode ask --slm qwen3.5-2b "$@"
