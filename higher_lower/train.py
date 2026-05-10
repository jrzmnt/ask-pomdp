"""
Train PPO on HigherLower (POPGym).

Usage:
  python higher_lower/train.py
  python higher_lower/train.py --timesteps 1000000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from ask.utils.seed import set_seed
from ask.utils.ppo import DropoutActorCriticPolicy
from higher_lower.env import HigherLowerEnv


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps, unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def make_env(seed: int = 0):
    env = HigherLowerEnv(num_decks=1)
    return Monitor(env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/higher_lower/model")
    args = p.parse_args()

    set_seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with wandb.init(
        project="ask-pomdp",
        name="train_higher_lower",
        group="higher_lower",
        job_type="train",
        config={
            "env": "HigherLowerEasy",
            "timesteps": args.timesteps,
            "seed": args.seed,
            "net_arch": [128, 128],
            "dropout_rate": 0.2,
        },
    ):
        env = make_env(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            DropoutActorCriticPolicy,
            env,
            policy_kwargs={"net_arch": [128, 128], "dropout_rate": 0.2},
            verbose=0,
            seed=args.seed,
            device=device,
        )
        model.learn(total_timesteps=args.timesteps, callback=TqdmCallback(args.timesteps))
        model.save(str(out))
        print(f"Model saved → {out}.zip")
        wandb.run.summary["model_path"] = str(out)


if __name__ == "__main__":
    main()
