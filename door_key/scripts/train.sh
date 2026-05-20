#!/usr/bin/env bash
# Train PPO on DoorKey. Pass --size 5|6|8|16 and any other args.
# Example: bash door_key/scripts/train.sh --size 5 --timesteps 500000
python door_key/train.py "$@"
