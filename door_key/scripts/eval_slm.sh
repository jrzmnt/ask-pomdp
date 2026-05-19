#!/usr/bin/env bash
# Evaluate SLM-only on DoorKey.
# Example: bash door_key/scripts/eval_slm.sh --size 5
python door_key/eval.py --mode slm --slm qwen3.5-2b "$@"
