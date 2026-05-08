from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
import yaml
from stable_baselines3 import PPO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

console = Console()

from ask.envs.fourrooms import FourRoomsEnv
from ask.utils.callbacks import EvalCallbackWithEvalMode
from ask.utils.ppo import DropoutActorCriticPolicy
from ask.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rl/ppo.yaml",
                   help="Path to YAML config (default: configs/rl/ppo.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    torch.manual_seed(cfg["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["experiment"]["seed"])

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="cyan")
    t.add_column(style="green")
    t.add_row("env",       cfg["env"]["name"])
    t.add_row("steps",     f"{cfg['training']['total_timesteps']:,}")
    t.add_row("net_arch",  str(cfg["policy"]["net_arch"]))
    t.add_row("dropout",   str(cfg["policy"]["dropout_rate"]))
    t.add_row("device",    cfg.get("device", "auto"))
    console.print(Panel(t, title="[bold]PPO Training[/bold]", border_style="cyan"))

    run = wandb.init(
        project="ask-pomdp",
        name="ppo_train",
        config=cfg,
        sync_tensorboard=True,
        save_code=False,
    )

    env = Monitor(FourRoomsEnv(max_steps=cfg["env"]["max_steps"]))
    eval_env = Monitor(FourRoomsEnv(max_steps=cfg["env"]["max_steps"]))

    model = PPO(
        policy=DropoutActorCriticPolicy,
        env=env,
        policy_kwargs={
            "dropout_rate": cfg["policy"]["dropout_rate"],
            "net_arch": cfg["policy"]["net_arch"],
        },
        n_steps=cfg["training"]["n_steps"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        tensorboard_log=f"runs/ppo/tb/{run.id}",
        verbose=0,
        device=cfg.get("device", "auto"),
    )

    callbacks = [
        EvalCallbackWithEvalMode(
            eval_env,
            best_model_save_path="./runs/ppo/best_model/",
            log_path="./runs/ppo/logs/",
            eval_freq=cfg["training"]["eval_freq"],
            n_eval_episodes=cfg["training"]["n_eval_episodes"],
            deterministic=True,
            render=False,
            verbose=0,
        ),
        WandbCallback(gradient_save_freq=0, verbose=0),
        ProgressBarCallback(),
    ]

    model.learn(total_timesteps=cfg["training"]["total_timesteps"], callback=callbacks)

    model_dir = Path("runs/ppo")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model")
    console.print(Panel(f"[green]Saved → {model_dir / 'model.zip'}[/green]", border_style="green"))

    run.finish()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
