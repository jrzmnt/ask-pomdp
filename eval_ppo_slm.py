"""
Evaluation script: PPO-only, SLM-only, and ASK (gated) on MiniGrid-FourRooms.

Metrics (per episode, then aggregated):
  - reward  : episode return
  - length  : steps until termination
  - IR      : Intervention Rate  = slm_called / steps  (fraction)
  - OR      : Overwrite Rate     = slm_overwrites / steps  (fraction)

Usage examples:
  python eval_ppo_slm.py                              # full pipeline, both SLMs, Optuna
  python eval_ppo_slm.py --mode ppo                   # PPO baseline only
  python eval_ppo_slm.py --mode slm --slm 1.5b        # SLM-only, 1.5B model
  python eval_ppo_slm.py --mode ask --slm 1.5b        # ASK with Optuna
  python eval_ppo_slm.py --mode ask --threshold 0.8   # ASK with fixed τ (skip Optuna)
  python eval_ppo_slm.py --mode ask --n-mc 10 --tag mc10   # ablation: N MC samples
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import wandb
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from stable_baselines3 import PPO
from tqdm import tqdm

from ask.envs.fourrooms import DIR_TO_STR, FourRoomsEnv
from ask.slm.model import load_slm
from ask.uncertainty.entropy import compute_mc_uncertainties
from ask.utils.seed import set_seed

console = Console()


# =============================================================================
# Constants
# =============================================================================

WANDB_PROJECT = "ask-pomdp"

QWEN_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

DECODING = {"max_tokens": 15}

N_EVAL_EPISODES = 100
N_TEST_EPISODES = 200
N_MC_SAMPLES = 30

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ACTIONS_STR = ["TURN_LEFT", "TURN_RIGHT", "FORWARD"]

# =============================================================================
# Prompt
# =============================================================================

PROMPT_TEMPLATE = """\
You are an agent in a partially observable grid world navigating to goal G.

Your 7×7 egocentric view (you are at row 6, col 3, facing toward row 0):
{grid}

Facing: {direction}
PPO autopilot suggests: {ppo_action}

Legend: A=you  .=floor  #=wall  D=door  G=goal  ?=unseen

Actions: TURN_LEFT  TURN_RIGHT  FORWARD
- FORWARD: move one step in the direction you are facing
- TURN_LEFT / TURN_RIGHT: rotate 90° in place
- You cannot pass through walls (#) or closed doors (D)

If you see the goal (G), navigate toward it.
If blocked ahead, turn to find an open path.
Explore unseen areas (?).

Output EXACTLY one word: TURN_LEFT or TURN_RIGHT or FORWARD\
"""


def build_prompt(env: FourRoomsEnv, ppo_action: int) -> str:
    grid = env.render_view_ascii()
    direction = DIR_TO_STR.get(env.agent_dir, "UNKNOWN")
    ppo_str = ACTIONS_STR[ppo_action] if ppo_action < 3 else "UNKNOWN"
    return PROMPT_TEMPLATE.format(grid=grid, direction=direction, ppo_action=ppo_str)


# =============================================================================
# Action parsing
# =============================================================================

_STR_TO_ACTION = {"TURN_LEFT": 0, "LEFT": 0, "TURN_RIGHT": 1, "RIGHT": 1, "FORWARD": 2, "UP": 2}


def parse_action(text: str) -> int | None:
    text = text.strip().upper()
    for key, val in _STR_TO_ACTION.items():
        if key in text:
            return val
    return None


# =============================================================================
# Helpers
# =============================================================================

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def short_model_name(model_name: str) -> str:
    name = model_name.lower()
    for tag in ["0.5b", "1.5b", "3b", "7b"]:
        if tag in name:
            return f"qwen_{tag}"
    return "qwen_unknown"


def resolve_model_path(override: str | None) -> str:
    if override:
        return override
    for candidate in ["runs/ppo/model", "runs/ppo/best_model/best_model"]:
        if Path(f"{candidate}.zip").exists():
            return candidate
    raise FileNotFoundError("No trained model found. Run train_ppo.py first.")


def slm_cfg_for(model_name: str) -> Dict[str, Any]:
    return {
        "provider": "hf",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16",
    }


# =============================================================================
# Shared: aggregate per-episode logs into a summary dict
# =============================================================================

def _summarize(logs: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compute paper metrics from a list of per-episode log dicts."""
    rewards  = [l["reward"] for l in logs]
    lengths  = [l["length"] for l in logs]
    successes = [l["reward"] > 0 for l in logs]
    lengths_success = [l["length"] for l in logs if l["reward"] > 0]

    summary: Dict[str, Any] = {
        "n_episodes":          len(logs),
        "mean_reward":         float(np.mean(rewards)),
        "std_reward":          float(np.std(rewards)),
        "success_rate":        float(np.mean(successes)),
        "mean_length":         float(np.mean(lengths)),
        "std_length":          float(np.std(lengths)),
        "mean_length_success": float(np.mean(lengths_success)) if lengths_success else float("nan"),
    }
    if extra:
        summary.update(extra)
    return summary


def _print_summary_table(title: str, summary: Dict[str, Any]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, min_width=40)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    for k, v in summary.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}" if not (v != v) else "—")  # nan → —
        else:
            table.add_row(k, str(v))
    console.print(table)


# =============================================================================
# PPO-only evaluation
# =============================================================================

def eval_ppo(
    model_path: str, n_episodes: int, seed_offset: int = 0
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = FourRoomsEnv()
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    model.policy.set_training_mode(False)

    logs = []
    for ep in tqdm(range(n_episodes), desc="PPO eval", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_reward, ep_len = False, 0.0, 0
        t0 = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        logs.append({
            "episode":        ep + 1,
            "reward":         ep_reward,
            "length":         ep_len,
            "result":         "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "episode_time_s": time.time() - t0,
        })

    env.close()
    return _summarize(logs), logs


# =============================================================================
# SLM-only evaluation
# =============================================================================

def eval_slm_only(
    slm_cfg: Dict[str, Any], n_episodes: int, seed_offset: int = 0
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = FourRoomsEnv()
    slm = load_slm(slm_cfg)

    model_tag = short_model_name(slm_cfg["model"])
    logs = []
    for ep in tqdm(range(n_episodes), desc=f"SLM {model_tag}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_reward, ep_len = False, 0.0, 0
        invalid_actions = 0
        t0 = time.time()
        while not done:
            prompt = build_prompt(env, ppo_action=2)
            output = slm.generate(prompt, DECODING)
            action = parse_action(output.text)
            if action is None:
                invalid_actions += 1
                action = 2  # fallback: FORWARD
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        logs.append({
            "episode":              ep + 1,
            "reward":               ep_reward,
            "length":               ep_len,
            "result":               "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "invalid_action_rate":  invalid_actions / ep_len if ep_len > 0 else 0.0,
            "episode_time_s":       time.time() - t0,
        })

    env.close()
    del slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extra = {
        "slm_model":          slm_cfg["model"],
        "invalid_action_rate": float(np.mean([l["invalid_action_rate"] for l in logs])),
    }
    return _summarize(logs, extra), logs


# =============================================================================
# ASK (gated) evaluation — core loop
# =============================================================================

def eval_ask(
    model: PPO,
    slm,
    threshold: float,
    n_episodes: int,
    seed_offset: int = 0,
    n_mc_samples: int = N_MC_SAMPLES,
) -> tuple[float, List[Dict[str, Any]]]:
    env = FourRoomsEnv()
    logs: List[Dict[str, Any]] = []

    for ep in tqdm(range(n_episodes), desc=f"ASK τ={threshold:.2f}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, ep_reward, steps = False, 0.0, 0
        slm_called, slm_valid, slm_overwrites = 0, 0, 0

        t0 = time.time()
        while not done:
            action_arr, _ = model.predict(obs, deterministic=True)
            ppo_action = int(action_arr)

            total_unc, _, _, _ = compute_mc_uncertainties(model, obs, n_samples=n_mc_samples)

            if total_unc >= threshold:
                slm_called += 1
                prompt = build_prompt(env, ppo_action)
                output = slm.generate(prompt, DECODING)
                slm_action = parse_action(output.text)

                if slm_action is not None:
                    slm_valid += 1
                    if slm_action != ppo_action:
                        ppo_action = slm_action
                        slm_overwrites += 1

            obs, reward, terminated, truncated, _ = env.step(ppo_action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        logs.append({
            "episode":         ep + 1,
            "reward":          ep_reward,
            "length":          steps,
            "result":          "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "IR":              slm_called / steps if steps > 0 else 0.0,
            "OR":              slm_overwrites / steps if steps > 0 else 0.0,
            "slm_valid_rate":  slm_valid / steps if steps > 0 else 0.0,
            "episode_time_s":  time.time() - t0,
        })

    env.close()
    return float(np.mean([l["reward"] for l in logs])), logs


# =============================================================================
# Optuna objective
# =============================================================================

def objective(
    trial: optuna.Trial,
    model_path: str,
    slm_cfg: Dict[str, Any],
    n_eval_episodes: int,
    n_mc_samples: int,
) -> float:
    threshold = trial.suggest_float("threshold", 0.1, 2.0)

    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    slm = load_slm(slm_cfg)

    mean_reward, _ = eval_ask(
        model=model, slm=slm, threshold=threshold,
        n_episodes=n_eval_episodes, seed_offset=0, n_mc_samples=n_mc_samples,
    )

    if mean_reward >= 0.999:
        trial.study.stop()

    del model, slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_reward


# =============================================================================
# Logging
# =============================================================================

def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_summary(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def wandb_log_summary(run, data: Dict[str, Any]) -> None:
    run.summary.update(data)
    run.log(data)


def wandb_log_episodes(run, logs: List[Dict[str, Any]]) -> None:
    table = wandb.Table(
        columns=list(logs[0].keys()),
        data=[[row[k] for k in logs[0].keys()] for row in logs],
    )
    run.log({"episodes": table})


def wandb_log_optuna_trials(run, study: "optuna.Study") -> None:
    rows = [
        {
            "trial":     t.number,
            "threshold": t.params["threshold"],
            "reward":    t.value,
            "state":     str(t.state),
        }
        for t in study.trials
        if t.value is not None
    ]
    if not rows:
        return
    table = wandb.Table(
        columns=list(rows[0].keys()),
        data=[[r[k] for k in rows[0].keys()] for r in rows],
    )
    run.log({"optuna_trials": table})


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO / SLM / ASK on MiniGrid-FourRooms")
    p.add_argument("--mode", choices=["ppo", "slm", "ask", "all"], default="all")
    p.add_argument("--slm", choices=["0.5b", "1.5b", "all"], default="all")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed τ — skips Optuna")
    p.add_argument("--n-mc", type=int, default=N_MC_SAMPLES, dest="n_mc")
    p.add_argument("--n-episodes", type=int, default=N_TEST_EPISODES, dest="n_episodes")
    p.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES, dest="n_eval_episodes")
    p.add_argument("--n-optuna-trials", type=int, default=15, dest="n_optuna_trials")
    p.add_argument("--model-path", type=str, default=None, dest="model_path")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix for output files, e.g. 'mc10'")
    p.add_argument("--wandb-group", type=str, default=None, dest="wandb_group",
                   help="W&B run group (e.g. 'main', 'ablation_threshold')")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(42)
    set_torch_seed(42)

    model_path = resolve_model_path(args.model_path)
    file_tag = f"_{args.tag}" if args.tag else ""
    slm_keys = ["0.5b", "1.5b"] if args.slm == "all" else [args.slm]

    # Determine W&B group and job_type from context
    is_ablation = bool(args.tag)
    group = args.wandb_group or ("ablation" if is_ablation else "main")
    job_type_ask = "ablation" if is_ablation else "eval_ask"

    # -------------------------------------------------------------------------
    # PPO-only baseline
    # -------------------------------------------------------------------------
    if args.mode in ("ppo", "all"):
        console.rule("[bold cyan]PPO baseline[/bold cyan]")
        with wandb.init(
            project=WANDB_PROJECT,
            name=f"eval_ppo{file_tag}",
            group=group,
            job_type="eval_ppo",
            config={
                "experiment": "ppo_baseline",
                "env": "MiniGrid-FourRooms-v0",
                "n_episodes": args.n_episodes,
            },
        ):
            summary, logs = eval_ppo(model_path, n_episodes=args.n_episodes, seed_offset=N_EVAL_EPISODES)
            _print_summary_table("PPO results", summary)
            wandb_log_summary(wandb.run, summary)
            wandb_log_episodes(wandb.run, logs)
            save_summary(summary, RESULTS_DIR / f"ppo_results{file_tag}.json")
            save_csv(logs, RESULTS_DIR / f"ppo_episodes{file_tag}.csv")

    # -------------------------------------------------------------------------
    # Per-model: SLM-only and/or ASK
    # -------------------------------------------------------------------------
    for key in slm_keys:
        model_name = QWEN_MODELS[key]
        tag = short_model_name(model_name)
        cfg = slm_cfg_for(model_name)

        # --- SLM-only ---
        if args.mode in ("slm", "all"):
            console.rule(f"[bold cyan]SLM-only — {tag}[/bold cyan]")
            with wandb.init(
                project=WANDB_PROJECT,
                name=f"eval_slm_{tag}{file_tag}",
                group=group,
                job_type="eval_slm",
                config={
                    "experiment": "slm_baseline",
                    "env": "MiniGrid-FourRooms-v0",
                    "slm_model": model_name,
                    "n_episodes": args.n_episodes,
                },
            ):
                summary, logs = eval_slm_only(cfg, n_episodes=args.n_episodes, seed_offset=N_EVAL_EPISODES)
                _print_summary_table(f"SLM {tag} results", summary)
                wandb_log_summary(wandb.run, summary)
                wandb_log_episodes(wandb.run, logs)
                save_summary(summary, RESULTS_DIR / f"slm_{tag}_results{file_tag}.json")
                save_csv(logs, RESULTS_DIR / f"slm_{tag}_episodes{file_tag}.csv")

        # --- ASK ---
        if args.mode in ("ask", "all"):
            wandb_cfg = {
                "experiment": f"ask_{args.tag}" if args.tag else "ask_main",
                "env": "MiniGrid-FourRooms-v0",
                "slm_model": model_name,
                "n_mc_samples": args.n_mc,
                "n_episodes": args.n_episodes,
                "n_eval_episodes": args.n_eval_episodes,
            }

            study = None
            if args.threshold is not None:
                best_threshold = args.threshold
                console.rule(f"[bold cyan]ASK — {tag}  τ={best_threshold:.4f} (fixed)[/bold cyan]")
                wandb_cfg["threshold"] = best_threshold
                wandb_cfg["threshold_source"] = "fixed"
            else:
                console.rule(f"[bold cyan]ASK — {tag}  Optuna ({args.n_optuna_trials} trials)[/bold cyan]")
                study_name = f"ask_{tag}{file_tag}"
                study = optuna.create_study(
                    direction="maximize",
                    storage="sqlite:///optuna.db",
                    study_name=study_name,
                    load_if_exists=True,
                )
                study.optimize(
                    lambda t: objective(t, model_path, cfg, args.n_eval_episodes, args.n_mc),
                    n_trials=args.n_optuna_trials,
                    show_progress_bar=True,
                )
                best_threshold = study.best_params["threshold"]
                console.print(
                    Panel(
                        f"Best τ = [green]{best_threshold:.4f}[/green]  |  "
                        f"eval reward = [green]{study.best_value:.4f}[/green]",
                        title="Optuna result", border_style="cyan",
                    )
                )
                wandb_cfg["threshold"] = best_threshold
                wandb_cfg["threshold_source"] = "optuna"
                wandb_cfg["optuna_best_reward"] = study.best_value
                wandb_cfg["optuna_n_trials"] = args.n_optuna_trials

            with wandb.init(
                project=WANDB_PROJECT,
                name=f"eval_ask_{tag}{file_tag}",
                group=group,
                job_type=job_type_ask,
                config=wandb_cfg,
            ):
                model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
                slm = load_slm(cfg)

                _, logs = eval_ask(
                    model=model, slm=slm, threshold=best_threshold,
                    n_episodes=args.n_episodes, seed_offset=args.n_eval_episodes,
                    n_mc_samples=args.n_mc,
                )

                del model, slm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                summary = _summarize(logs, extra={
                    "slm_model":     model_name,
                    "threshold":     best_threshold,
                    "n_mc_samples":  args.n_mc,
                    "IR_mean":       float(np.mean([l["IR"] for l in logs])),
                    "IR_std":        float(np.std([l["IR"] for l in logs])),
                    "OR_mean":       float(np.mean([l["OR"] for l in logs])),
                    "OR_std":        float(np.std([l["OR"] for l in logs])),
                    "slm_valid_rate": float(np.mean([l["slm_valid_rate"] for l in logs])),
                })
                _print_summary_table(f"ASK {tag} results", summary)
                wandb_log_summary(wandb.run, summary)
                wandb_log_episodes(wandb.run, logs)
                if study is not None:
                    wandb_log_optuna_trials(wandb.run, study)
                save_summary(summary, RESULTS_DIR / f"ask_{tag}_results{file_tag}.json")
                save_csv(logs, RESULTS_DIR / f"ask_{tag}_episodes{file_tag}.csv")


if __name__ == "__main__":
    main()
