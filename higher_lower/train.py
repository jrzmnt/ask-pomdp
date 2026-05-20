"""
Train PPO on HigherLower (POPGym) with reward-threshold checkpoints.

Saves model_reward_NNN.zip the first time eval reward crosses each threshold,
giving qualitatively different PPO agents for the optimality ablation.

Usage:
  python higher_lower/train.py
  python higher_lower/train.py --timesteps 500000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor

from ask.utils.seed import set_seed
from ask.utils.ppo import DropoutActorCriticPolicy
from ask.utils.callbacks import RewardThresholdCheckpointCallback
from higher_lower.env import HigherLowerEnv


# Reward thresholds for checkpointing.
# HigherLower PPO typically reaches ~0.49 at convergence;
# these 4 levels cover near-random → near-optimal.
CHECKPOINT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]


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


def make_env():
    return Monitor(HigherLowerEnv(num_decks=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/higher_lower/model")
    args = p.parse_args()

    set_seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(
        project="ask-pomdp-v2",
        name="hl_train",
        group="higherlower",
        job_type="train",
        config={
            "env": "HigherLowerEasy",
            "timesteps": args.timesteps,
            "checkpoint_thresholds": CHECKPOINT_THRESHOLDS,
            "seed": args.seed,
            "net_arch": [128, 128],
            "dropout_rate": 0.2,
        },
    ):
        env = make_env()
        eval_env = make_env()
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
                best_model_save_path=str(out.parent / "best_model"),
                log_path=str(out.parent / "logs"),
                eval_freq=5_000,
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
