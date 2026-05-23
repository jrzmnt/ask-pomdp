#!/usr/bin/env python3
"""Generate paper figures from existing ASK results.

Outputs go to ``plots/figures/{fig1,fig2,fig3}.{pdf,png}`` (one figure per
relationship, one panel per environment).

Usage::

    python plots/make_figures.py all
    python plots/make_figures.py fig1   # τ sensitivity (Optuna trials)
    python plots/make_figures.py fig2   # N_MC sensitivity
    python plots/make_figures.py fig3   # PPO checkpoint sweep

Data sources:
  * Optuna SQLite (``optuna.db``)        — Fig 1 reward curves
  * ``*/results/ask_*_threshold_*.json`` — Fig 1 dense IR curves (optional)
  * ``*/results/ask_*_mc*.json``         — Fig 2
  * ``*/results/ask_*_ckpt_*.json``      — Fig 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "plots" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_DB = ROOT / "optuna.db"

# --------------------------------------------------------------------------- #
# Environment registry
# --------------------------------------------------------------------------- #

C_2B = "#1f77b4"
C_4B = "#d62728"
C_RND = "#7f7f7f"


def _fr_ckpt(c: str) -> str:
    return f"ask_qwen35_{{model_tag}}_results_ckpt_r{c}.json"


ENVS: List[Dict] = [
    {
        "key": "fourrooms",
        "label": "FourRooms",
        "results_dir": ROOT / "results",
        "thresholds_file": ROOT / "results" / "thresholds.json",
        "reward_label": "Reward",
        "models": {
            "Qwen3.5-2B": {
                "study": "ask_qwen35_2b",
                "thr_key": "fourrooms_qwen35_2b",
                "results": "ask_qwen35_2b_results.json",
                "ckpt_pattern": "ask_qwen35_2b_results_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_2b_results_mc{n}.json",
                "thr_pattern": "ask_qwen35_2b_results_threshold_{tau}.json",
                "color": C_2B,
            },
            "Qwen3.5-4B": {
                "study": "ask_qwen35_4b",
                "thr_key": "fourrooms_qwen35_4b",
                "results": "ask_qwen35_4b_results.json",
                "ckpt_pattern": "ask_qwen35_4b_results_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_4b_results_mc{n}.json",
                "thr_pattern": "ask_qwen35_4b_results_threshold_{tau}.json",
                "color": C_4B,
            },
            # "random": {
            #     "study": "ask_random",
            #     "thr_key": "fourrooms_random",
            #     "results": "ask_random_results.json",
            #     "ckpt_pattern": None,
            #     "mc_pattern": "ask_random_results_mc{n}.json",
            #     "thr_pattern": "ask_random_results_threshold_{tau}.json",
            #     "color": C_RND,
            # },
        },
        "ckpts": ["010", "030", "050"],
        "ppo_results": "ppo_results.json",
        "ppo_ckpt_pattern": "ppo_results_ckpt_r{ckpt}.json",
    },
    {
        "key": "higherlower",
        "label": "HigherLower",
        "results_dir": ROOT / "higher_lower" / "results",
        "thresholds_file": ROOT / "higher_lower" / "results" / "thresholds.json",
        "reward_label": "Reward",
        "models": {
            "Qwen3.5-2B": {
                "study": "hl_ask_qwen35_2b",
                "thr_key": "higherlower_qwen35_2b",
                "results": "ask_qwen35_2b_results.json",
                "ckpt_pattern": "ask_qwen35_2b_results_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_2b_results_mc{n}.json",
                "thr_pattern": "ask_qwen35_2b_results_threshold_{tau}.json",
                "color": C_2B,
            },
            "Qwen3.5-4B": {
                "study": "hl_ask_qwen35_4b",
                "thr_key": "higherlower_qwen35_4b",
                "results": "ask_qwen35_4b_results.json",
                "ckpt_pattern": "ask_qwen35_4b_results_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_4b_results_mc{n}.json",
                "thr_pattern": "ask_qwen35_4b_results_threshold_{tau}.json",
                "color": C_4B,
            },
            # "random": {
            #     "study": "hl_ask_random",
            #     "thr_key": "higherlower_random",
            #     "results": "ask_random_results.json",
            #     "ckpt_pattern": None,
            #     "mc_pattern": "ask_random_results_mc{n}.json",
            #     "thr_pattern": "ask_random_results_threshold_{tau}.json",
            #     "color": C_RND,
            # },
        },
        "ckpts": ["010", "020", "030", "040"],
        "ppo_results": "ppo_results.json",
        "ppo_ckpt_pattern": "ppo_results_ckpt_r{ckpt}.json",
    },
    {
        "key": "doorkey",
        "label": "DoorKey-8x8",
        "results_dir": ROOT / "door_key" / "results",
        "thresholds_file": ROOT / "door_key" / "results" / "thresholds.json",
        "reward_label": "Reward",
        "models": {
            "Qwen3.5-2B": {
                "study": "dk_ask_s8_qwen35_2b",
                "thr_key": "doorkey_s8_qwen35_2b",
                "results": "ask_qwen35_2b_results_s8.json",
                "ckpt_pattern": "ask_qwen35_2b_results_s8_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_2b_results_s8_mc{n}.json",
                "thr_pattern": "ask_qwen35_2b_results_s8_threshold_{tau}.json",
                "color": C_2B,
            },
            "Qwen3.5-4B": {
                "study": "dk_ask_s8_qwen35_4b",
                "thr_key": "doorkey_s8_qwen35_4b",
                "results": "ask_qwen35_4b_results_s8.json",
                "ckpt_pattern": "ask_qwen35_4b_results_s8_ckpt_r{ckpt}.json",
                "mc_pattern": "ask_qwen35_4b_results_s8_mc{n}.json",
                "thr_pattern": "ask_qwen35_4b_results_s8_threshold_{tau}.json",
                "color": C_4B,
            },
            # "random": {
            #     "study": "dk_ask_s8_random",
            #     "thr_key": "doorkey_s8_random",
            #     "results": "ask_random_results_s8.json",
            #     "ckpt_pattern": None,
            #     "mc_pattern": "ask_random_results_s8_mc{n}.json",
            #     "thr_pattern": "ask_random_results_s8_threshold_{tau}.json",
            #     "color": C_RND,
            # },
        },
        "ckpts": ["030", "050", "070"],
        "ppo_results": "ppo_results_s8.json",
        "ppo_ckpt_pattern": "ppo_results_s8_ckpt_r{ckpt}.json",
    },
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,  # twinx will re-enable on its own axis
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def load_thresholds_registry(path: Path) -> Dict:
    return load_json(path) or {}


def optuna_trials(study_name: str) -> List[Tuple[float, float]]:
    """Return list of (threshold, value) for a study, sorted by threshold."""
    if not OPTUNA_DB.exists():
        return []
    con = sqlite3.connect(str(OPTUNA_DB))
    cur = con.cursor()
    cur.execute(
        """
        SELECT tp.param_value, tv.value
        FROM trials t
        JOIN trial_params tp ON tp.trial_id = t.trial_id AND tp.param_name = 'threshold'
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        JOIN studies s ON s.study_id = t.study_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE'
        ORDER BY tp.param_value
        """,
        (study_name,),
    )
    rows = cur.fetchall()
    con.close()
    return [(float(t), float(v)) for t, v in rows if v is not None]


def reward_key(payload: dict) -> Optional[float]:
    if payload is None:
        return None
    for k in ("mean_reward", "reward"):
        if k in payload:
            return float(payload[k])
    return None


def success_key(payload: dict) -> Optional[float]:
    if payload is None:
        return None
    for k in ("success_rate", "mean_success", "mean_accuracy"):
        if k in payload:
            return float(payload[k])
    return None


def ir_key(payload: dict) -> Optional[float]:
    if payload is None:
        return None
    for k in ("IR_mean", "ir_mean", "intervention_rate"):
        if k in payload:
            return float(payload[k])
    return None


def time_from_csv(csv_path: Path) -> Optional[Tuple[float, float]]:
    """Return (mean, std) of ``episode_time_s`` from a per-episode CSV, or None."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            vals = [float(row["episode_time_s"]) for row in reader if row.get("episode_time_s")]
    except Exception:
        return None
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def discover_sweep_files(results_dir: Path, pattern: str) -> List[Tuple[str, Path]]:
    """Find files matching ``pattern`` with the sweep variable as a glob."""
    sweep_var = "{tau}" if "{tau}" in pattern else "{n}" if "{n}" in pattern else None
    if sweep_var is None:
        return []
    glob = pattern.replace(sweep_var, "*")
    rgx = re.compile("^" + re.escape(pattern).replace(re.escape(sweep_var), "(.+?)") + "$")
    out: List[Tuple[str, Path]] = []
    for p in sorted(results_dir.glob(glob)):
        m = rgx.match(p.name)
        if m:
            out.append((m.group(1), p))
    return out


def parse_threshold_tag(s: str) -> Optional[float]:
    """Parse the tag suffix written by ``ablation_threshold.sh`` (``${TAU/./}``).

    Examples (digits only): ``"01" -> 0.1``, ``"10" -> 1.0``, ``"175" -> 1.75``,
    ``"005" -> 0.05``. Returns None when the tag isn't a pure digit string.
    """
    if not s.isdigit() or len(s) < 2:
        return None
    return float(s[0] + "." + s[1:])


def collect_checkpoint_tau_ir(env: Dict, mcfg: Dict) -> List[Tuple[float, float, Optional[float]]]:
    """Harvest (τ, IR, reward) tuples from per-checkpoint + full-PPO JSONs.

    Used as a fallback when no dense τ-sweep JSONs exist. The τ values come
    from each run's Optuna result (saved alongside ``IR_mean`` in the same
    JSON). Returns rows sorted by τ.
    """
    out: List[Tuple[float, float, Optional[float]]] = []
    rdir = env["results_dir"]
    candidates: List[Path] = [rdir / mcfg["results"]]
    if mcfg.get("ckpt_pattern"):
        for c in env["ckpts"]:
            candidates.append(rdir / mcfg["ckpt_pattern"].format(ckpt=c))
    for p in candidates:
        payload = load_json(p)
        if payload is None:
            continue
        tau = payload.get("threshold")
        ir = ir_key(payload)
        rew = reward_key(payload)
        if tau is None or ir is None:
            continue
        out.append((float(tau), float(ir), rew))
    out.sort(key=lambda r: r[0])
    return out


# --------------------------------------------------------------------------- #
# Fig 1: τ sensitivity
# --------------------------------------------------------------------------- #


def _panel_fig1(ax_left, env: Dict) -> Tuple[List[str], List]:
    """One panel of Fig 1 (reward + IR vs τ). Returns (notes, reward_handles)."""
    ax_right = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    thr_registry = load_thresholds_registry(env["thresholds_file"])

    reward_handles: List = []
    notes: List[str] = []

    for label, mcfg in env["models"].items():
        trials = optuna_trials(mcfg["study"])
        color = mcfg["color"]

        # ----- Dense τ sweep JSONs (optional but preferred) -----
        sweep_files = discover_sweep_files(env["results_dir"], mcfg["thr_pattern"])
        sweep_pts: List[Tuple[float, float, float]] = []  # (τ, reward, IR)
        for tau_str, path in sweep_files:
            payload = load_json(path)
            if payload is None:
                continue
            tau = parse_threshold_tag(tau_str)
            if tau is None:
                try:
                    tau = float(tau_str)
                except ValueError:
                    continue
            r = reward_key(payload)
            ir = ir_key(payload)
            if r is None or ir is None:
                continue
            sweep_pts.append((tau, r, ir))
        sweep_pts.sort()

        # ----- Reward curve -----
        if sweep_pts:
            taus = [p[0] for p in sweep_pts]
            rs = [p[1] for p in sweep_pts]
            h, = ax_left.plot(taus, rs, color=color, marker="o", lw=1.6,
                              label=label)
            reward_handles.append(h)
        elif trials:
            taus = [t for t, _ in trials]
            rs = [r for _, r in trials]
            ax_left.scatter(taus, rs, color=color, marker="o", s=22, alpha=0.4)
            order = np.argsort(taus)
            taus_s = np.asarray(taus)[order]
            rs_s = np.asarray(rs)[order]
            if len(rs_s) >= 3:
                pad = max(3, len(rs_s) // 4)
                kernel = np.ones(pad) / pad
                rs_smooth = np.convolve(rs_s, kernel, mode="same")
                h, = ax_left.plot(taus_s, rs_smooth, color=color, lw=1.6, label=label)
            else:
                h, = ax_left.plot(taus_s, rs_s, color=color, lw=1.4, marker="o", label=label)
            reward_handles.append(h)

        # ----- Optuna-selected τ -----
        thr_entry = thr_registry.get(mcfg["thr_key"])
        best_tau = float(thr_entry["threshold"]) if thr_entry is not None else None
        if best_tau is not None:
            ax_left.axvline(best_tau, color=color, ls=":", lw=1.0, alpha=0.6)

        # ----- IR curve (right axis) -----
        if sweep_pts:
            taus = [p[0] for p in sweep_pts]
            irs = [p[2] for p in sweep_pts]
            ax_right.plot(taus, irs, color=color, ls="--", lw=1.4,
                          marker="s", ms=4, alpha=0.85)
        else:
            # Fallback: collect (τ, IR) from per-ckpt + full evals. These mix
            # different PPO qualities but give a real, monotone-ish curve.
            ck_pts = collect_checkpoint_tau_ir(env, mcfg)
            if len(ck_pts) >= 2:
                taus_c = [p[0] for p in ck_pts]
                irs_c = [p[1] for p in ck_pts]
                ax_right.plot(taus_c, irs_c, color=color, ls="--", lw=1.4,
                              marker="s", ms=4, alpha=0.85)
                notes.append(f"IR ({label}): combines full+ckpt evals; PPO quality varies")
            elif best_tau is not None:
                payload = load_json(env["results_dir"] / mcfg["results"])
                ir_pt = ir_key(payload)
                if ir_pt is not None:
                    ax_right.scatter([best_tau], [ir_pt], color=color,
                                     marker="s", s=36, edgecolor="k",
                                     linewidth=0.6, alpha=0.9, zorder=5)

    ax_left.set_xlabel(r"Uncertainty threshold $\tau$")
    ax_left.set_ylabel("Reward (solid)")
    ax_right.set_ylabel("Intervention rate (dashed)")
    ax_right.set_ylim(-0.05, 1.05)
    ax_left.set_title(env["label"])
    return notes, reward_handles


def fig1() -> Path:
    configure_style()
    fig, axes = plt.subplots(1, len(ENVS), figsize=(13.5, 4.0))
    if len(ENVS) == 1:
        axes = [axes]
    all_notes: List[str] = []
    legend_pool: "Dict[str, object]" = {}  # label → handle (de-duplicated)
    for ax, env in zip(axes, ENVS):
        notes, handles = _panel_fig1(ax, env)
        all_notes.extend(notes)
        for h in handles:
            lbl = h.get_label()
            if lbl and lbl not in legend_pool:
                legend_pool[lbl] = h
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))  # reserve bottom strip for legend
    if legend_pool:
        fig.legend(
            handles=list(legend_pool.values()),
            labels=list(legend_pool.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncols=len(legend_pool),
            frameon=False,
        )
    out_pdf = FIG_DIR / "fig1_threshold.pdf"
    out_png = FIG_DIR / "fig1_threshold.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    if all_notes:
        print(
            "[fig1] No dense τ sweep found; IR curves were built from per-checkpoint evals "
            "(mixed PPO qualities). Run `bash scripts/ablation_threshold_all.sh` for a clean sweep:"
        )
        for n in all_notes:
            print(f"        {n}")
    return out_png


# --------------------------------------------------------------------------- #
# Fig 2: N_MC sensitivity
# --------------------------------------------------------------------------- #


def _collect_mc_points(env: Dict, mcfg: Dict) -> List[Tuple[int, dict, Optional[Tuple[float, float]]]]:
    """Return [(N, summary_json, (mean_time, std_time) or None), ...] sorted by N."""
    sweep = discover_sweep_files(env["results_dir"], mcfg["mc_pattern"])
    pts: List[Tuple[int, dict, Optional[Tuple[float, float]]]] = []
    for n_str, path in sweep:
        try:
            n = int(n_str)
        except ValueError:
            continue
        payload = load_json(path)
        if payload is None:
            continue
        # Sibling CSV holds per-episode times
        csv_path = path.with_name(path.name.replace("_results_", "_episodes_").replace(".json", ".csv"))
        times = time_from_csv(csv_path)
        pts.append((n, payload, times))
    pts.sort(key=lambda x: x[0])
    return pts


def _panel_fig2(ax_left, env: Dict, missing: List[str]) -> None:
    ax_right = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    any_data = False
    handles_r: List = []
    handles_t: List = []
    for label, mcfg in env["models"].items():
        if label == "random":
            continue  # skip dice in MC sweep
        pts = _collect_mc_points(env, mcfg)
        if not pts:
            missing.append(f"{env['key']}/{label}: {mcfg['mc_pattern']}")
            continue
        any_data = True
        ns = np.array([n for n, _, _ in pts])
        rews = np.array([reward_key(p) for _, p, _ in pts])
        # std reward
        stds = np.array([float(p.get("std_reward", 0.0)) for _, p, _ in pts])
        # times
        ts = np.array([(t[0] if t else np.nan) for _, _, t in pts])
        ts_std = np.array([(t[1] if t else 0.0) for _, _, t in pts])
        color = mcfg["color"]
        h, _, _ = ax_left.errorbar(ns, rews, yerr=stds, color=color, marker="o",
                                   capsize=2.5, lw=1.6, label=f"{label}")
        handles_r.append(h)
        if not np.all(np.isnan(ts)):
            h_t = ax_right.errorbar(ns, ts, yerr=ts_std, color=color, marker="s",
                                    capsize=2.5, lw=1.2, ls="--", alpha=0.9,
                                    label=f"{label} time")[0]
            handles_t.append(h_t)

    if not any_data:
        ax_left.text(0.5, 0.5,
                     "No MC sweep data yet — run\n  bash scripts/ablation_mc_samples_all.sh",
                     ha="center", va="center", transform=ax_left.transAxes,
                     fontsize=9, alpha=0.7)
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_right.set_yticks([])
    else:
        ax_left.set_xlabel("MC Dropout samples $N$")
        ax_left.set_ylabel("Reward (solid)")
        ax_right.set_ylabel("Per-episode time, s (dashed)")
        if handles_r:
            ax_left.legend(handles=handles_r, loc="lower right", frameon=False)
    ax_left.set_title(env["label"])


def fig2() -> Path:
    configure_style()
    missing: List[str] = []
    fig, axes = plt.subplots(1, len(ENVS), figsize=(13.5, 3.8))
    if len(ENVS) == 1:
        axes = [axes]
    for ax, env in zip(axes, ENVS):
        _panel_fig2(ax, env, missing)
    fig.suptitle(
        "Fig. 2 — Effect of MC Dropout samples $N$ on reward (solid) "
        "and per-episode wall-clock time (dashed).",
        y=1.02, fontsize=10,
    )
    out_pdf = FIG_DIR / "fig2_mc_samples.pdf"
    out_png = FIG_DIR / "fig2_mc_samples.png"
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    if missing:
        print(
            "[fig2] No MC-sweep data found for:\n  - "
            + "\n  - ".join(missing)
            + "\n  Run `bash scripts/ablation_mc_samples_all.sh` to populate."
        )
    return out_png


# --------------------------------------------------------------------------- #
# Fig 3: PPO quality sweep
# --------------------------------------------------------------------------- #


def _panel_fig3(ax_left, env: Dict) -> None:
    ax_right = ax_left.twinx()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    rdir = env["results_dir"]
    # Collect PPO baseline points: (ppo_reward, ask_reward, IR) per ckpt + full
    x_full = None
    ppo_full = load_json(rdir / env["ppo_results"])
    if ppo_full is not None:
        x_full = reward_key(ppo_full)

    handles: List = []
    for label, mcfg in env["models"].items():
        if mcfg["ckpt_pattern"] is None:
            continue
        xs: List[float] = []
        ys: List[float] = []
        irs: List[float] = []
        # checkpoints
        for c in env["ckpts"]:
            ppo_json = load_json(rdir / env["ppo_ckpt_pattern"].format(ckpt=c))
            ask_json = load_json(rdir / mcfg["ckpt_pattern"].format(ckpt=c))
            ppo_r = reward_key(ppo_json)
            ask_r = reward_key(ask_json)
            ask_ir = ir_key(ask_json)
            if ppo_r is None or ask_r is None:
                continue
            xs.append(ppo_r)
            ys.append(ask_r)
            irs.append(ask_ir if ask_ir is not None else np.nan)
        # full model
        full_ask = load_json(rdir / mcfg["results"])
        full_r = reward_key(full_ask)
        full_ir = ir_key(full_ask)
        if x_full is not None and full_r is not None:
            xs.append(x_full)
            ys.append(full_r)
            irs.append(full_ir if full_ir is not None else np.nan)

        if not xs:
            continue
        # sort by ppo reward
        order = np.argsort(xs)
        xs_s = np.asarray(xs)[order]
        ys_s = np.asarray(ys)[order]
        irs_s = np.asarray(irs)[order]

        color = mcfg["color"]
        h, = ax_left.plot(xs_s, ys_s, color=color, marker="o", lw=1.6,
                          label=label)
        handles.append(h)
        ax_right.plot(xs_s, irs_s, color=color, ls="--", marker="s", lw=1.2,
                      alpha=0.85)

    # Diagonal: ASK == PPO
    if x_full is not None:
        lo, hi = ax_left.get_xlim()
        diag_lo = min(0.0, lo)
        diag_hi = max(1.0, hi, x_full + 0.05)
        ax_left.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="0.7",
                     lw=0.8, ls="-", alpha=0.7, zorder=0)
        ax_left.set_xlim(diag_lo, diag_hi)

    ax_left.set_xlabel("PPO reward (per checkpoint)")
    ax_left.set_ylabel("ASK reward (solid)")
    ax_right.set_ylabel("Intervention rate (dashed)")
    ax_right.set_ylim(-0.05, 1.05)
    ax_left.set_title(env["label"])
    if handles:
        ax_left.legend(handles=handles, loc="lower right", frameon=False)


def fig3() -> Path:
    configure_style()
    fig, axes = plt.subplots(1, len(ENVS), figsize=(13.5, 3.8))
    if len(ENVS) == 1:
        axes = [axes]
    for ax, env in zip(axes, ENVS):
        _panel_fig3(ax, env)
    fig.suptitle(
        "Fig. 3 — ASK reward (solid) and intervention rate (dashed) "
        "vs. underlying PPO policy quality. Grey diagonal: ASK == PPO.",
        y=1.02, fontsize=10,
    )
    out_pdf = FIG_DIR / "fig3_ppo_quality.pdf"
    out_png = FIG_DIR / "fig3_ppo_quality.png"
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


CMDS: Dict[str, Callable[[], Path]] = {
    "fig1": fig1,
    "fig2": fig2,
    "fig3": fig3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="?", default="all",
                        choices=["fig1", "fig2", "fig3", "all"])
    args = parser.parse_args()
    targets = list(CMDS) if args.which == "all" else [args.which]
    for name in targets:
        try:
            path = CMDS[name]()
            print(f"[ok] {name} → {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[err] {name} failed: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
