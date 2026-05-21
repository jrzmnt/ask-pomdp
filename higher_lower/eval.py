"""
Evaluate PPO / SLM-only / ASK on HigherLower (POPGym).

Usage:
  python higher_lower/eval.py --mode ppo
  python higher_lower/eval.py --mode slm  --slm 1.5b
  python higher_lower/eval.py --mode ask  --slm 1.5b
  python higher_lower/eval.py --mode ask  --slm 1.5b --threshold 0.8
  python higher_lower/eval.py --mode slm --slm qwen3.5-2b --prompt-style stateful --prompt-rationale
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from ask.slm.model import load_slm
from ask.uncertainty.entropy import compute_mc_uncertainties
from ask.utils.seed import set_seed
from higher_lower.env import HigherLowerEnv

console = Console()

WANDB_PROJECT = "ask-pomdp-v2"

QWEN_MODELS = {
    "0.5b":         "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b":         "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":   "Qwen/Qwen3-1.7B",
    "qwen3.5-2b":   "Qwen/Qwen3.5-2B",
    "qwen3.5-4b":   "Qwen/Qwen3.5-4B",
}

DECODING = {"max_tokens": 10}
DECODING_RATIONALE_MAX_TOKENS = 32

N_EVAL_EPISODES = 100
N_TEST_EPISODES = 100
N_MC_SAMPLES = 30

RESULTS_DIR = Path("higher_lower/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Match longer tokens first when scanning (substring match)
_PARSE_ACTION_ORDER = [
    ("HIGHER", 0),
    ("LOWER", 1),
    ("HIGH", 0),
    ("LOW", 1),
]

ACTIONS_STR = ["HIGHER", "LOWER"]


def decoding_for(rationale: bool) -> Dict[str, Any]:
    d = dict(DECODING)
    if rationale:
        d["max_tokens"] = DECODING_RATIONALE_MAX_TOKENS
    return d


# =============================================================================
# Helpers
# =============================================================================

def short_model_name(model_name: str) -> str:
    if model_name == "random":
        return "random"
    name = model_name.lower().replace("/", "-").replace("_", "-").replace(".", "")
    for pattern, tag in [
        ("qwen35-4b",  "qwen35_4b"),
        ("qwen35-2b",  "qwen35_2b"),
        ("qwen3-06b",  "qwen3_0.6b"),
        ("qwen3-17b",  "qwen3_1.7b"),
        ("qwen25-05b", "qwen25_0.5b"),
        ("qwen25-15b", "qwen25_1.5b"),
        ("05b",        "qwen25_0.5b"),
        ("15b",        "qwen25_1.5b"),
    ]:
        if pattern in name:
            return tag
    return "qwen_unknown"


def slm_cfg_for(model_name: str) -> Dict[str, Any]:
    if model_name == "random":
        return {
            "provider": "random",
            "model": "random",
            "actions": list(ACTIONS_STR),
            "seed": 42,
        }
    return {
        "provider": "hf",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16",
    }


def parse_action(text: str, *, rationale: bool = False) -> Optional[int]:
    t = text.strip()
    if rationale:
        for line in t.splitlines():
            s = line.strip()
            if s.upper().startswith("ACTION:"):
                tail = s.split(":", 1)[-1].strip().upper()
                for key, val in _PARSE_ACTION_ORDER:
                    if key in tail:
                        return val
    u = t.upper()
    for key, val in _PARSE_ACTION_ORDER:
        if key in u:
            return val
    return None


def _summarize(logs: List[Dict], extra: Optional[Dict] = None) -> Dict:
    rewards = [l["reward"] for l in logs]
    accs = [l["accuracy"] for l in logs]
    summary: Dict[str, Any] = {
        "n_episodes":   len(logs),
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
    }
    if extra:
        summary.update(extra)
    return summary


def _print_summary_table(title: str, summary: Dict) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, min_width=40)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    for k, v in summary.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}" if v == v else "—")
        else:
            table.add_row(k, str(v))
    console.print(table)


def save_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_summary(data: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"  Saved → {path}")


def save_threshold(key: str, entry: Dict[str, Any]) -> None:
    """Persist Optuna threshold under a structured key.

    Key format:
      main run       → "higherlower_{model_tag}"
      ckpt ablation  → "higherlower_{model_tag}_{ckpt_tag}"
    """
    path = RESULTS_DIR / "thresholds.json"
    registry: Dict = {}
    if path.exists():
        with open(path) as f:
            registry = json.load(f)
    registry[key] = entry
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    console.print(f"  Threshold saved → {path} [{key}]")


def wandb_log_episodes(run, logs: List[Dict]) -> None:
    table = wandb.Table(
        columns=list(logs[0].keys()),
        data=[[row[k] for k in logs[0].keys()] for row in logs],
    )
    run.log({"episodes": table})


def wandb_log_optuna_trials(run, study: "optuna.Study") -> None:
    rows = [
        {"trial": t.number, "threshold": t.params["threshold"], "reward": t.value, "state": str(t.state)}
        for t in study.trials if t.value is not None
    ]
    if not rows:
        return
    table = wandb.Table(
        columns=list(rows[0].keys()),
        data=[[r[k] for k in rows[0].keys()] for r in rows],
    )
    run.log({"optuna_trials": table})


# =============================================================================
# PPO eval
# =============================================================================

def _obs_arr(obs: int) -> np.ndarray:
    return np.array([obs], dtype=np.float32)


def eval_ppo(model_path: str, n_episodes: int, seed_offset: int = 0):
    env = HigherLowerEnv()
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    model.policy.set_training_mode(False)

    logs = []
    for ep in tqdm(range(n_episodes), desc="PPO eval", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, correct, total_steps = False, 0.0, 0, 0
        t0 = time.time()
        while not done:
            action, _ = model.predict(_obs_arr(obs), deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(int(action.flat[0]))
            total_reward += float(reward)
            if reward > 0:
                correct += 1
            total_steps += 1
            obs = next_obs
            done = terminated or truncated
        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward, "accuracy": correct / total_steps if total_steps else 0.0,
            "steps": total_steps, "IR": 0.0, "OR": 0.0,
            "slm_valid_rate": 0.0, "invalid_action_rate": 0.0,
            "episode_time_s": time.time() - t0,
        })
    env.close()
    return _summarize(logs), logs


# =============================================================================
# SLM-only eval
# =============================================================================

def eval_slm_only(
    slm_cfg: Dict,
    n_episodes: int,
    seed_offset: int = 0,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
):
    env = HigherLowerEnv()
    slm = load_slm(slm_cfg)
    tag = short_model_name(slm_cfg["model"])
    decode = decoding_for(prompt_rationale)

    logs = []
    for ep in tqdm(range(n_episodes), desc=f"SLM {tag}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, correct, total_steps, invalid = False, 0.0, 0, 0, 0
        t0 = time.time()
        while not done:
            prompt = env.build_prompt(
                None,
                prompt_style=prompt_style,
                rationale=prompt_rationale,
                prompt_history=prompt_history,
            )
            output = slm.generate(prompt, decode)
            action = parse_action(output.text, rationale=prompt_rationale)
            if action is None:
                invalid += 1
                action = 0  # fallback: HIGHER
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if reward > 0:
                correct += 1
            total_steps += 1
            obs = next_obs
            done = terminated or truncated
        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward,
            "accuracy": correct / total_steps if total_steps else 0.0,
            "steps": total_steps, "IR": 1.0, "OR": float("nan"),
            "slm_valid_rate": 1.0 - invalid / total_steps if total_steps else 0.0,
            "invalid_action_rate": invalid / total_steps if total_steps else 0.0,
            "episode_time_s": time.time() - t0,
        })
    env.close()
    del slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _summarize(
        logs,
        {
            "slm_model": slm_cfg["model"],
            "prompt_style": prompt_style,
            "prompt_rationale": prompt_rationale,
            "prompt_history": prompt_history,
        },
    ), logs


# =============================================================================
# ASK eval
# =============================================================================

def eval_ask(
    model: PPO,
    slm,
    threshold: float,
    n_episodes: int,
    seed_offset: int = 0,
    n_mc_samples: int = N_MC_SAMPLES,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
):
    env = HigherLowerEnv()
    logs = []
    decode = decoding_for(prompt_rationale)

    for ep in tqdm(range(n_episodes), desc=f"ASK τ={threshold:.2f}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, correct, steps = False, 0.0, 0, 0
        slm_called, slm_valid, slm_overwrites, slm_invalid = 0, 0, 0, 0
        t0 = time.time()
        while not done:
            obs_arr = _obs_arr(obs)
            action_arr, _ = model.predict(obs_arr, deterministic=True)
            ppo_action = int(action_arr.flat[0])

            total_unc, _, _, _ = compute_mc_uncertainties(model, obs_arr, n_samples=n_mc_samples)

            if total_unc >= threshold:
                slm_called += 1
                prompt = env.build_prompt(
                    ppo_action,
                    prompt_style=prompt_style,
                    rationale=prompt_rationale,
                    prompt_history=prompt_history,
                )
                output = slm.generate(prompt, decode)
                slm_action = parse_action(output.text, rationale=prompt_rationale)
                if slm_action is not None:
                    slm_valid += 1
                    if slm_action != ppo_action:
                        ppo_action = slm_action
                        slm_overwrites += 1
                else:
                    slm_invalid += 1

            next_obs, reward, terminated, truncated, _ = env.step(ppo_action)
            total_reward += float(reward)
            if reward > 0:
                correct += 1
            steps += 1
            obs = next_obs
            done = terminated or truncated

        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward,
            "accuracy": correct / steps if steps else 0.0,
            "steps": steps,
            "IR": slm_called / steps if steps else 0.0,
            "OR": slm_overwrites / steps if steps else 0.0,
            "slm_valid_rate": slm_valid / slm_called if slm_called > 0 else 0.0,
            "invalid_action_rate": slm_invalid / slm_called if slm_called > 0 else 0.0,
            "episode_time_s": time.time() - t0,
        })
    env.close()
    return float(np.mean([l["reward"] for l in logs])), logs


def objective(
    trial,
    model,
    slm,
    n_eval_episodes,
    n_mc_samples,
    *,
    prompt_style: str,
    prompt_rationale: bool,
    prompt_history: int,
):
    threshold = trial.suggest_float("threshold", 0.01, 1.5)
    mean_reward, _ = eval_ask(
        model,
        slm,
        threshold,
        n_eval_episodes,
        seed_offset=0,
        n_mc_samples=n_mc_samples,
        prompt_style=prompt_style,
        prompt_rationale=prompt_rationale,
        prompt_history=prompt_history,
    )
    return mean_reward


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ppo", "slm", "ask"], default="ppo")
    p.add_argument(
        "--slm",
        choices=list(QWEN_MODELS.keys()) + ["random"],
        default="qwen3.5-2b",
        help='Qwen tag or "random" (dice baseline)',
    )
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--n-mc", type=int, default=N_MC_SAMPLES, dest="n_mc")
    p.add_argument("--n-episodes", type=int, default=N_TEST_EPISODES, dest="n_episodes")
    p.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES, dest="n_eval_episodes")
    p.add_argument("--n-optuna-trials", type=int, default=15, dest="n_optuna_trials")
    p.add_argument("--model-path", type=str, default="runs/higher_lower/model", dest="model_path")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="higherlower", dest="wandb_group")
    p.add_argument(
        "--prompt-style",
        choices=["basic", "enriched", "stateful"],
        default="basic",
        dest="prompt_style",
        help="SLM prompt: basic (legacy), enriched (+probabilities), stateful (+episode memory)",
    )
    p.add_argument(
        "--prompt-rationale",
        action="store_true",
        dest="prompt_rationale",
        help="Allow Reason: line before Action: (increases max_new_tokens)",
    )
    p.add_argument(
        "--prompt-history",
        type=int,
        default=8,
        dest="prompt_history",
        help="Recent decisions shown in stateful prompts",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    torch.manual_seed(42)

    file_tag = f"_{args.tag}" if args.tag else ""

    # Extract checkpoint reward from tag (e.g. "ckpt_r030" → 0.30)
    checkpoint_reward = None
    if args.tag.startswith("ckpt_r"):
        try:
            checkpoint_reward = int(args.tag[6:]) / 100.0
        except ValueError:
            pass

    if args.mode == "ppo":
        console.rule("[bold cyan]PPO — HigherLower[/bold cyan]")
        cfg_ppo = {"env": "HigherLowerEasy", "n_episodes": args.n_episodes}
        if checkpoint_reward is not None:
            cfg_ppo["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"hl_eval_ppo{file_tag}",
                        group=args.wandb_group, job_type="eval_ppo", config=cfg_ppo):
            summary, logs = eval_ppo(args.model_path, args.n_episodes, seed_offset=N_EVAL_EPISODES)
            _print_summary_table("PPO results", summary)
            if checkpoint_reward is not None:
                summary["checkpoint_reward"] = checkpoint_reward
            wandb.run.summary.update(summary)
            save_summary(summary, RESULTS_DIR / f"ppo_results{file_tag}.json")
            save_csv(logs, RESULTS_DIR / f"ppo_episodes{file_tag}.csv")

    elif args.mode == "slm":
        model_name = "random" if args.slm == "random" else QWEN_MODELS[args.slm]
        tag = short_model_name(model_name)
        cfg = slm_cfg_for(model_name)
        console.rule(f"[bold cyan]SLM-only — {tag} — HigherLower[/bold cyan]")
        cfg_slm = {
            "env": "HigherLowerEasy",
            "slm_model": model_name,
            "n_episodes": args.n_episodes,
            "prompt_style": args.prompt_style,
            "prompt_rationale": args.prompt_rationale,
            "prompt_history": args.prompt_history,
        }
        if checkpoint_reward is not None:
            cfg_slm["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"hl_eval_slm_{tag}{file_tag}",
                        group=args.wandb_group, job_type="eval_slm", config=cfg_slm):
            summary, logs = eval_slm_only(
                cfg,
                args.n_episodes,
                seed_offset=N_EVAL_EPISODES,
                prompt_style=args.prompt_style,
                prompt_rationale=args.prompt_rationale,
                prompt_history=args.prompt_history,
            )
            _print_summary_table(f"SLM {tag} results", summary)
            if checkpoint_reward is not None:
                summary["checkpoint_reward"] = checkpoint_reward
            wandb.run.summary.update(summary)
            save_summary(summary, RESULTS_DIR / f"slm_{tag}_results{file_tag}.json")
            save_csv(logs, RESULTS_DIR / f"slm_{tag}_episodes{file_tag}.csv")

    elif args.mode == "ask":
        model_name = "random" if args.slm == "random" else QWEN_MODELS[args.slm]
        tag = short_model_name(model_name)
        cfg = slm_cfg_for(model_name)

        study = None
        if args.threshold is not None:
            best_threshold = args.threshold
            console.rule(f"[bold cyan]ASK — {tag} τ={best_threshold:.4f} (fixed)[/bold cyan]")
            _opt_model = PPO.load(args.model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            _opt_slm = load_slm(cfg)
        else:
            console.rule(f"[bold cyan]ASK — {tag} Optuna ({args.n_optuna_trials} trials)[/bold cyan]")
            study_name = f"hl_ask_{tag}{file_tag}"
            study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db",
                                        study_name=study_name, load_if_exists=True)
            # Load model + SLM once — reused across all trials and final eval
            _opt_model = PPO.load(args.model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            _opt_slm = load_slm(cfg)
            study.optimize(
                lambda t: objective(
                    t,
                    _opt_model,
                    _opt_slm,
                    args.n_eval_episodes,
                    args.n_mc,
                    prompt_style=args.prompt_style,
                    prompt_rationale=args.prompt_rationale,
                    prompt_history=args.prompt_history,
                ),
                n_trials=args.n_optuna_trials, show_progress_bar=True,
            )
            best_threshold = study.best_params["threshold"]
            console.print(Panel(
                f"Best τ = [green]{best_threshold:.4f}[/green]  |  "
                f"eval reward = [green]{study.best_value:.4f}[/green]",
                title="Optuna result", border_style="cyan",
            ))
            threshold_key = f"higherlower_{tag}" + (f"_{args.tag}" if args.tag else "")
            save_threshold(threshold_key, {
                "threshold":       best_threshold,
                "optuna_study":    study_name,
                "eval_reward":     study.best_value,
                "model_path":      args.model_path,
                "slm_model":       model_name,
                "n_trials":        args.n_optuna_trials,
                "n_mc_samples":    args.n_mc,
                "n_eval_episodes": args.n_eval_episodes,
                "env":             "HigherLowerEasy",
                "saved_at":        datetime.now().isoformat(),
            })

        cfg_ask = {
            "env": "HigherLowerEasy",
            "slm_model": model_name,
            "threshold": best_threshold,
            "n_mc_samples": args.n_mc,
            "n_episodes": args.n_episodes,
            "prompt_style": args.prompt_style,
            "prompt_rationale": args.prompt_rationale,
            "prompt_history": args.prompt_history,
        }
        if checkpoint_reward is not None:
            cfg_ask["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"hl_eval_ask_{tag}{file_tag}",
                        group=args.wandb_group, job_type="eval_ask", config=cfg_ask):
            _, logs = eval_ask(
                _opt_model,
                _opt_slm,
                best_threshold,
                args.n_episodes,
                seed_offset=N_EVAL_EPISODES,
                n_mc_samples=args.n_mc,
                prompt_style=args.prompt_style,
                prompt_rationale=args.prompt_rationale,
                prompt_history=args.prompt_history,
            )
            del _opt_model, _opt_slm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            summary = _summarize(logs, {
                "slm_model": model_name,
                "threshold": best_threshold,
                "n_mc_samples": args.n_mc,
                "prompt_style": args.prompt_style,
                "prompt_rationale": args.prompt_rationale,
                "prompt_history": args.prompt_history,
                "IR_mean": float(np.mean([l["IR"] for l in logs])),
                "OR_mean": float(np.mean([l["OR"] for l in logs])),
                "slm_valid_rate": float(np.mean([l["slm_valid_rate"] for l in logs])),
                "invalid_action_rate": float(np.mean([l["invalid_action_rate"] for l in logs])),
            })
            if checkpoint_reward is not None:
                summary["checkpoint_reward"] = checkpoint_reward
            _print_summary_table(f"ASK {tag} results", summary)
            wandb.run.summary.update(summary)
            save_summary(summary, RESULTS_DIR / f"ask_{tag}_results{file_tag}.json")
            save_csv(logs, RESULTS_DIR / f"ask_{tag}_episodes{file_tag}.csv")


if __name__ == "__main__":
    main()
