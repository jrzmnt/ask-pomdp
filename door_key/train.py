"""
Train PPO on MiniGrid DoorKey with reward-threshold checkpoints.

Saves model_reward_NNN.zip when eval reward first crosses each threshold,
giving qualitatively different PPO agents for the optimality ablation.

Usage:
  python door_key/train.py --size 5
  python door_key/train.py --size 8 --timesteps 1000000
"""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ask.utils.callbacks import RewardThresholdCheckpointCallback
from ask.utils.ppo import DropoutActorCriticPolicy
from ask.utils.seed import set_seed
from door_key.env import SeededDoorKeyEnv, TRAIN_SEEDS, VAL_SEEDS

# Reward thresholds. DoorKey max reward ≈ 1.0 (perfect play, step 1/max_steps).
CHECKPOINT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        from tqdm import tqdm
        self.pbar = tqdm(total=total_timesteps, unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=5, choices=[5, 6, 8, 16])
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)

    if args.out is None:
        args.out = f"runs/door_key/model_s{args.size}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out.parent / f"checkpoints_s{args.size}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_name = f"DoorKey-{args.size}x{args.size}"

    with wandb.init(
        project="ask-pomdp-v2",
        name=f"dk_train_s{args.size}",
        group="doorkey",
        job_type="train",
        config={
            "env": env_name,
            "size": args.size,
            "timesteps": args.timesteps,
            "checkpoint_thresholds": CHECKPOINT_THRESHOLDS,
            "seed": args.seed,
            "net_arch": [128, 128],
            "dropout_rate": 0.2,
        },
    ):
        # Enforce seed splits: train on maps 200-999, eval callback on maps 0-99.
        env = Monitor(SeededDoorKeyEnv(size=args.size, seeds=TRAIN_SEEDS))
        eval_env = Monitor(SeededDoorKeyEnv(size=args.size, seeds=VAL_SEEDS))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            DropoutActorCriticPolicy,
            env,
            policy_kwargs={"net_arch": [128, 128], "dropout_rate": 0.2},
            verbose=0,
            seed=args.seed,
            device=device,
        )

        callbacks = [
            RewardThresholdCheckpointCallback(
                thresholds=CHECKPOINT_THRESHOLDS,
                checkpoint_dir=str(checkpoint_dir),
                eval_env=eval_env,
                best_model_save_path=str(out.parent / f"best_model_s{args.size}"),
                log_path=str(out.parent / "logs"),
                eval_freq=10_000,
                n_eval_episodes=50,
                deterministic=True,
                render=False,
                verbose=0,
            ),
            ProgressCallback(args.timesteps),
        ]

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )

        model.save(str(out))
        print(f"Final model saved → {out}.zip")
        wandb.run.summary["model_path"] = str(out)

        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
