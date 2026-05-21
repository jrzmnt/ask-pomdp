"""
Evaluate PPO / SLM-only / ASK on MiniGrid DoorKey.

Usage:
  python door_key/eval.py --mode ppo --size 5
  python door_key/eval.py --mode slm --slm qwen3.5-2b --size 5
  python door_key/eval.py --mode ask --slm qwen3.5-2b --size 5
  python door_key/eval.py --mode ask --slm qwen3.5-2b --size 5 --threshold 0.8
  python door_key/eval.py --mode slm --slm qwen3.5-2b --size 5 --prompt-style stateful --prompt-rationale
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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import wandb
from minigrid.core.constants import OBJECT_TO_IDX
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from stable_baselines3 import PPO
from tqdm import tqdm

from ask.slm.model import load_slm
from ask.uncertainty.entropy import compute_mc_uncertainties
from ask.utils.seed import set_seed
from door_key.env import (
    ACTIONS,
    DIR_TO_STR,
    DOOR_STATE_CHAR,
    DoorKeyEnv,
    OBJECT_TO_CHAR,
    TEST_SEEDS,
    VAL_SEEDS,
    _STR_TO_ACTION,
)

console = Console()

WANDB_PROJECT = "ask-pomdp-v2"

QWEN_MODELS = {
    "0.5b":        "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b":        "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b":  "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":  "Qwen/Qwen3-1.7B",
    "qwen3.5-2b":  "Qwen/Qwen3.5-2B",
    "qwen3.5-4b":  "Qwen/Qwen3.5-4B",
}

DECODING = {"max_tokens": 10}
DECODING_RATIONALE_MAX_TOKENS = 48

N_EVAL_EPISODES = len(VAL_SEEDS)   # 100 — Optuna τ search (seeds 0-99)
N_TEST_EPISODES = len(TEST_SEEDS)  # 100 — final reported results (seeds 100-199)
N_MC_SAMPLES = 30

RESULTS_DIR = Path("door_key/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Short letters used in recent-actions trace
ACTION_LETTER = {0: "L", 1: "R", 2: "F", 3: "P", 4: "D", 5: "T", 6: "E"}

# MiniGrid agent_dir → unit forward vector (x, y)
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# Order matters for substring match (longer tokens first)
_PARSE_ACTION_ORDER: List[Tuple[str, int]] = sorted(
    _STR_TO_ACTION.items(), key=lambda kv: len(kv[0]), reverse=True
)


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
            "actions": list(ACTIONS),
            "seed": 42,
        }
    return {
        "provider": "hf",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16",
    }


def decoding_for(rationale: bool) -> Dict[str, Any]:
    d = dict(DECODING)
    if rationale:
        d["max_tokens"] = DECODING_RATIONALE_MAX_TOKENS
    return d


def _parse_action_substring(text_upper: str) -> Optional[int]:
    for key, val in _PARSE_ACTION_ORDER:
        if key in text_upper:
            return val
    return None


def parse_action(text: str, *, rationale: bool = False) -> Optional[int]:
    """Parse an SLM response into a DoorKey action id.

    When ``rationale=True``, the line beginning with ``Action:`` is checked first
    so that words like ``TOGGLE`` appearing in the Reason line don't override
    the actual final decision.
    """
    t = text.strip()
    if rationale:
        for line in t.splitlines():
            s = line.strip()
            if s.upper().startswith("ACTION:"):
                tail = s.split(":", 1)[-1].strip().upper()
                hit = _parse_action_substring(tail)
                if hit is not None:
                    return hit
    return _parse_action_substring(t.upper())


# =============================================================================
# Episode state + prompt enrichment
# =============================================================================


@dataclass
class EpisodeState:
    visits: Dict[Tuple[int, int], int] = field(default_factory=dict)
    known_grid: Dict[Tuple[int, int], str] = field(default_factory=dict)
    actions: List[int] = field(default_factory=list)
    key_pos_abs: Optional[Tuple[int, int]] = None
    door_pos_abs: Optional[Tuple[int, int]] = None
    goal_pos_abs: Optional[Tuple[int, int]] = None
    last_pos: Optional[Tuple[int, int]] = None
    stuck_steps: int = 0
    had_key_step: Optional[int] = None
    door_open_step: Optional[int] = None
    recent_pos_hist: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=5))


def new_episode_state() -> EpisodeState:
    return EpisodeState()


def _merge_known_cell(old: str, new: str) -> str:
    """Combine the previously known char at a cell with a fresh observation."""
    if new == "?" or new == " ":
        return old
    # Goal beats everything else (rare to be replaced)
    if new == "G":
        return "G"
    if old == "G":
        return "G"
    # Walls are immutable
    if new == "#":
        return "#"
    # Door state transitions: locked -> closed -> open
    if old in ("L", "D", "o") and new in ("L", "D", "o"):
        order = {"L": 0, "D": 1, "o": 2}
        return new if order[new] >= order[old] else old
    if new in ("L", "D", "o"):
        return new
    # Key disappears once picked up: don't downgrade k to . using stale views;
    # always trust freshest non-key observation when it's "."
    return new


def merge_obs_into_known_grid(state: EpisodeState, env: DoorKeyEnv) -> None:
    raw = env.raw_obs
    if raw is None:
        return
    img = raw["image"]
    for r in range(7):
        for c in range(7):
            obj_type = int(img[r, c, 0])
            if obj_type == 4:
                ch = DOOR_STATE_CHAR.get(int(img[r, c, 2]), "D")
            else:
                ch = OBJECT_TO_CHAR.get(obj_type, "?")
            if ch in ("?", " "):
                continue
            pos = env.obs_cell_to_world(r, c)
            prev = state.known_grid.get(pos, "?")
            state.known_grid[pos] = _merge_known_cell(prev, ch)
            if ch == "G":
                state.goal_pos_abs = pos
            elif ch == "k":
                state.key_pos_abs = pos
            elif ch in ("L", "D", "o"):
                state.door_pos_abs = pos


def update_episode_state(
    state: EpisodeState, env: DoorKeyEnv, action_taken: Optional[int]
) -> None:
    merge_obs_into_known_grid(state, env)
    pos = env.agent_pos_abs
    prev = state.last_pos
    if prev is not None and prev == pos and (action_taken is None or action_taken == 2):
        state.stuck_steps += 1
    else:
        state.stuck_steps = 0
    state.last_pos = pos
    state.visits[pos] = state.visits.get(pos, 0) + 1
    state.recent_pos_hist.append(pos)
    if action_taken is not None:
        state.actions.append(int(action_taken))
    # If the agent now holds the key, the previously cached key tile is empty.
    if env.has_key:
        if state.had_key_step is None:
            state.had_key_step = len(state.actions)
        if state.key_pos_abs is not None and state.key_pos_abs in state.known_grid:
            state.known_grid[state.key_pos_abs] = "."
        state.key_pos_abs = None
    if env.door_open and state.door_open_step is None:
        state.door_open_step = len(state.actions)


def _world_cell_char(env: DoorKeyEnv, wx: int, wy: int) -> str:
    g = env._env.unwrapped.grid
    if wx < 0 or wy < 0 or wx >= g.width or wy >= g.height:
        return "#"
    cell = g.get(wx, wy)
    if cell is None:
        return "."
    t = cell.type
    if isinstance(t, int):
        return OBJECT_TO_CHAR.get(t, ".")
    if isinstance(t, str):
        if t == "door":
            if getattr(cell, "is_open", False):
                return "o"
            if getattr(cell, "is_locked", False):
                return "L"
            return "D"
        idx = OBJECT_TO_IDX.get(t, 1)
        return OBJECT_TO_CHAR.get(idx, ".")
    return "."


def _corridor_blocked(ch: str) -> bool:
    """Cells that stop a free-space scan."""
    return ch in ("#", "?", "~", "L", "D")


def longest_corridor_world(env: DoorKeyEnv, pos: Tuple[int, int]) -> Tuple[str, int]:
    best_name, best_len = "FORWARD", 0
    for name, vec in (
        ("FORWARD", DIR_TO_VEC[env.agent_dir]),
        ("RIGHT", DIR_TO_VEC[(env.agent_dir + 1) % 4]),
        ("BACK", DIR_TO_VEC[(env.agent_dir + 2) % 4]),
        ("LEFT", DIR_TO_VEC[(env.agent_dir + 3) % 4]),
    ):
        dx, dy = vec
        x, y = pos
        n = 0
        while True:
            x += dx
            y += dy
            ch = _world_cell_char(env, x, y)
            if _corridor_blocked(ch):
                break
            n += 1
        if n > best_len:
            best_len = n
            best_name = name
    return best_name, best_len


def _world_delta_ego(
    agent_pos: Tuple[int, int], agent_dir: int, target: Tuple[int, int]
) -> Tuple[int, int]:
    """Express ``target - agent`` in (forward, right) steps in agent frame."""
    ax, ay = agent_pos
    gx, gy = target
    dgx, dgy = gx - ax, gy - ay
    fdx, fdy = DIR_TO_VEC[agent_dir]
    rdx, rdy = DIR_TO_VEC[(agent_dir + 1) % 4]
    along = dgx * fdx + dgy * fdy
    perp = dgx * rdx + dgy * rdy
    return along, perp


def _describe_bearing(along: int, perp: int) -> str:
    parts: List[str] = []
    if along > 0:
        parts.append(f"{along} forward")
    elif along < 0:
        parts.append(f"{abs(along)} behind")
    if perp > 0:
        parts.append(f"{perp} right")
    elif perp < 0:
        parts.append(f"{abs(perp)} left")
    if not parts:
        return "on agent tile"
    return ", ".join(parts)


def _parse_grid_features(env: DoorKeyEnv) -> Dict[str, Any]:
    """Navigation features from the current POV and world geometry."""
    raw = env.raw_obs
    if raw is None:
        return {
            "ahead": "?", "left": "?", "right": "?", "back": "?",
            "ahead_chain": ["?", "?", "?"],
            "visible": {}, "longest_corridor": ("forward_in_view", 0),
        }
    img = raw["image"]
    ax, ay = env.agent_pos_abs
    d = env.agent_dir
    fdx, fdy = DIR_TO_VEC[d]
    rdx, rdy = DIR_TO_VEC[(d + 1) % 4]
    ldx, ldy = -rdx, -rdy
    bdx, bdy = -fdx, -fdy

    def wc(dx: int, dy: int) -> str:
        return _world_cell_char(env, ax + dx, ay + dy)

    ahead = wc(fdx, fdy)
    left = wc(ldx, ldy)
    right = wc(rdx, rdy)
    back = wc(bdx, bdy)
    ahead_chain = [wc(fdx * k, fdy * k) for k in (1, 2, 3)]

    visible: Dict[str, List[str]] = {"k": [], "D": [], "L": [], "o": [], "G": []}
    ar, ac = _agent_image_ij(env)
    for r in range(7):
        for c in range(7):
            obj_type = int(img[r, c, 0])
            if obj_type == 4:
                ch = DOOR_STATE_CHAR.get(int(img[r, c, 2]), "D")
            else:
                ch = OBJECT_TO_CHAR.get(obj_type, "?")
            if ch not in visible:
                continue
            if (r, c) == (ar, ac):
                continue
            wx, wy = env.obs_cell_to_world(r, c)
            along, perp = _world_delta_ego((ax, ay), d, (wx, wy))
            visible[ch].append(_describe_bearing(along, perp))

    img_dirs = [
        ("forward_in_view", (-1, 0)),
        ("right_in_view", (0, 1)),
        ("back_in_view", (1, 0)),
        ("left_in_view", (0, -1)),
    ]
    best_name, best_len = "forward_in_view", 0
    for name, (dr, dc) in img_dirs:
        rr, cc = ar, ac
        n = 0
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < 7 and 0 <= cc < 7):
                break
            obj_type = int(img[rr, cc, 0])
            ch = OBJECT_TO_CHAR.get(obj_type, "?")
            if obj_type == 4 or ch in ("#", "?", " ", ""):
                break
            n += 1
        if n > best_len:
            best_len = n
            best_name = name

    return {
        "ahead": ahead,
        "left": left,
        "right": right,
        "back": back,
        "ahead_chain": ahead_chain,
        "visible": visible,
        "longest_corridor": (best_name, best_len),
    }


def _agent_image_ij(env: DoorKeyEnv) -> Tuple[int, int]:
    ap = env.agent_pos_abs
    for r in range(7):
        for c in range(7):
            if env.obs_cell_to_world(r, c) == ap:
                return r, c
    return 6, 3


def _action_previews(env: DoorKeyEnv, pos: Tuple[int, int], agent_dir: int) -> List[str]:
    """One-line consequence for each of the 7 DoorKey actions."""
    ax, ay = pos
    lines: List[str] = []

    nd = (agent_dir - 1) % 4
    vx, vy = DIR_TO_VEC[nd]
    lines.append(
        f"TURN_LEFT  → facing={DIR_TO_STR[nd]}, ahead would be {_world_cell_char(env, ax+vx, ay+vy)!r}"
    )
    nd = (agent_dir + 1) % 4
    vx, vy = DIR_TO_VEC[nd]
    lines.append(
        f"TURN_RIGHT → facing={DIR_TO_STR[nd]}, ahead would be {_world_cell_char(env, ax+vx, ay+vy)!r}"
    )
    vx, vy = DIR_TO_VEC[agent_dir]
    fx, fy = ax + vx, ay + vy
    fch = _world_cell_char(env, fx, fy)
    if fch in ("#", "L", "D"):
        lines.append(f"FORWARD    → blocked (ahead is {fch!r})")
    elif fch == "o":
        lines.append(f"FORWARD    → step through open door to ({fx},{fy})")
    else:
        lines.append(f"FORWARD    → move to ({fx},{fy}), cell is {fch!r}")

    if fch == "k":
        if env.has_key:
            lines.append("PICKUP     → already carrying an object (no-op)")
        else:
            lines.append("PICKUP     → would pick up the KEY in front")
    else:
        lines.append(f"PICKUP     → nothing pickable directly ahead (sees {fch!r})")

    if env.has_key:
        lines.append("DROP       → would drop the carried key on the empty tile in front (if empty)")
    else:
        lines.append("DROP       → nothing to drop (not carrying)")

    if fch == "L":
        if env.has_key:
            lines.append("TOGGLE     → would UNLOCK the door in front (you have the key)")
        else:
            lines.append("TOGGLE     → door is LOCKED but you do not hold the key")
    elif fch == "D":
        lines.append("TOGGLE     → would OPEN the closed door in front")
    elif fch == "o":
        lines.append("TOGGLE     → would CLOSE the open door in front (rarely useful)")
    else:
        lines.append(f"TOGGLE     → no door directly ahead (sees {fch!r})")

    lines.append("DONE       → ends the episode immediately; only use after reaching the goal G")
    return lines


def render_known_window(state: EpisodeState, env: DoorKeyEnv, radius: int) -> str:
    ax, ay = env.agent_pos_abs
    w, h = env._env.unwrapped.width, env._env.unwrapped.height
    rows: List[str] = []
    for yy in range(ay - radius, ay + radius + 1):
        buf: List[str] = []
        for xx in range(ax - radius, ax + radius + 1):
            if xx == ax and yy == ay:
                buf.append("A")
                continue
            if not (0 <= xx < w and 0 <= yy < h):
                buf.append("#")
                continue
            ch = state.known_grid.get((xx, yy), "?")
            buf.append(ch)
        rows.append("".join(buf))
    return "\n".join(rows)


def _format_recent_actions(actions: List[int], n: int) -> str:
    tail = actions[-n:] if n > 0 else []
    return " ".join(ACTION_LETTER.get(a, "?") for a in tail) if tail else "(none)"


def _visits_last_n(state: EpisodeState, n: int) -> str:
    seq = list(state.recent_pos_hist)[-n:]
    return str([state.visits.get(p, 0) for p in seq])


def _loop_warnings(state: EpisodeState) -> str:
    lines: List[str] = []
    if state.stuck_steps >= 6:
        lines.append(
            "LOOP_HINT: You have not moved to a new cell for several steps; "
            "prefer aligning with a clear corridor or interacting with key/door."
        )
    if len(state.actions) >= 5 and all(a in (0, 1) for a in state.actions[-5:]):
        lines.append("LOOP_HINT: Many consecutive turns; align with a clear corridor or cached target bearing.")
    return "\n".join(lines)


def _current_subtask(env: DoorKeyEnv) -> str:
    if env.door_open:
        return "REACH THE GREEN GOAL TILE (G); navigate to G then optionally DONE"
    if env.has_key:
        return "FIND AND UNLOCK THE DOOR (face door 'L' or 'D' and TOGGLE)"
    return "FIND AND PICK UP THE KEY ('k') — face it then PICKUP"


def _bfs_hint_to_target(
    env: DoorKeyEnv,
    state: EpisodeState,
    agent_pos: Tuple[int, int],
    agent_dir: int,
    target: Tuple[int, int],
    *,
    allow_door: bool = False,
) -> Optional[str]:
    """Return a short navigation hint toward ``target`` on the known map."""
    if agent_pos == target:
        return None
    known = state.known_grid
    blocked = {"#", "?", "~"}
    if not allow_door:
        blocked |= {"L", "D"}
    passable = {p for p, ch in known.items() if ch not in blocked}
    passable.add(agent_pos)
    passable.add(target)
    if target not in known:
        return None

    q: Deque[Tuple[int, int]] = deque([agent_pos])
    prev_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {agent_pos: None}
    reached = False
    while q:
        cur = q.popleft()
        if cur == target:
            reached = True
            break
        x, y = cur
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nxt = (nx, ny)
            if nxt not in passable or nxt in prev_map:
                continue
            prev_map[nxt] = cur
            q.append(nxt)
    if not reached:
        return None

    cur = target
    first_step: Optional[Tuple[int, int]] = None
    while cur != agent_pos:
        p = prev_map.get(cur)
        if p is None:
            return None
        if p == agent_pos:
            first_step = cur
            break
        cur = p
    if first_step is None:
        return None
    dx, dy = first_step[0] - agent_pos[0], first_step[1] - agent_pos[1]
    fdx, fdy = DIR_TO_VEC[agent_dir]
    rdx, rdy = DIR_TO_VEC[(agent_dir + 1) % 4]
    if (dx, dy) == (fdx, fdy):
        return "FORWARD (one step on known map toward target)"
    if (dx, dy) == (rdx, rdy):
        return "TURN_RIGHT then FORWARD (target is on your right)"
    if (dx, dy) == (-rdx, -rdy):
        return "TURN_LEFT then FORWARD (target is on your left)"
    return "TURN_LEFT twice (or RIGHT twice) to face target, then FORWARD"


def _current_target(env: DoorKeyEnv, state: EpisodeState) -> Tuple[Optional[Tuple[int, int]], bool]:
    """Pick the next sub-goal position and whether to treat doors as passable."""
    if env.door_open:
        return state.goal_pos_abs, True
    if env.has_key:
        return state.door_pos_abs, True
    return state.key_pos_abs, False


# =============================================================================
# Prompt construction
# =============================================================================


_PROMPT_STYLES = ("basic", "enriched", "stateful_min", "stateful")


def build_prompt(
    env: DoorKeyEnv,
    state: Optional[EpisodeState] = None,
    ppo_action: Optional[int] = None,
    *,
    prompt_style: str = "basic",
    rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
) -> str:
    if prompt_style not in _PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {_PROMPT_STYLES}, got {prompt_style!r}")

    grid = env.render_view_ascii()
    direction = DIR_TO_STR.get(env.agent_dir, "?")
    subtask = _current_subtask(env)
    has_key_str = "YES" if env.has_key else "NO"
    if env.door_open:
        door_str = "OPEN"
    elif env.door_locked:
        door_str = "LOCKED"
    else:
        door_str = "CLOSED"
    ppo_line = f"\nAutopilot suggests: {ACTIONS[int(ppo_action)]}" if ppo_action is not None else ""

    legend = (
        "LEGEND: A=agent, k=key, L=door(locked), D=door(closed), o=door(open), "
        "G=goal, #=wall, .=empty, ?=unknown"
    )

    if prompt_style == "basic":
        rules = (
            "- Do NOT explain.\n"
            "- Do NOT add text or markdown.\n"
            "- PICKUP: pick up the key when you face it.\n"
            "- TOGGLE: unlock/open the door when you face it (key required).\n"
            "- FORWARD: move one step in the direction you face.\n"
            "- Avoid walls (#). Navigate around them.\n"
        )
        out_fmt = "Action: "
        return f"""\
You are a navigation policy for a MiniGrid DoorKey task.
Your task is to choose exactly ONE action.

VALID ACTIONS:
TURN_LEFT
TURN_RIGHT
FORWARD
PICKUP
DROP
TOGGLE
DONE

RULES:
{rules}
{legend}

CURRENT VIEW (7×7 egocentric):
{grid}

STATUS:
Facing: {direction}
Has key: {has_key_str}
Door: {door_str}
Step: {env.step_count}/{env.max_steps}

CURRENT SUBTASK:
{subtask}{ppo_line}

{out_fmt}"""

    # ---- enriched / stateful* ----
    features = _parse_grid_features(env)
    chain_s = " ".join(features["ahead_chain"])
    lc_name, lc_len = features["longest_corridor"]
    pos = env.agent_pos_abs
    lcw_name, lcw_len = longest_corridor_world(env, pos)
    previews = _action_previews(env, pos, env.agent_dir)

    visible = features["visible"]
    def fmt_visible(label: str, key: str) -> str:
        items = visible.get(key, [])
        return f"{label}: " + (", ".join(items) if items else "(none in view)")

    enriched_block = f"""\
Adjacent (forward/left/right/back): {features['ahead']}/{features['left']}/{features['right']}/{features['back']}
Ahead chain (steps 1-3): {chain_s}
{fmt_visible('Key (k) visible', 'k')}
{fmt_visible('Door L locked', 'L')}
{fmt_visible('Door D closed', 'D')}
{fmt_visible('Door o open', 'o')}
{fmt_visible('Goal (G) visible', 'G')}
Longest clear ray in view: {lc_name} ({lc_len} steps)
Longest clear ray in world frame: {lcw_name} ({lcw_len} steps)

ACTION PREVIEW (ground truth from simulator):
{chr(10).join(previews)}
"""

    state_block = ""
    map_block = ""
    hint_line = ""
    if prompt_style in ("stateful_min", "stateful") and state is not None:
        cached_lines: List[str] = []
        for label, p in (
            ("Key cached at", state.key_pos_abs),
            ("Door cached at", state.door_pos_abs),
            ("Goal cached at", state.goal_pos_abs),
        ):
            if p is None:
                continue
            along, perp = _world_delta_ego(pos, env.agent_dir, p)
            cached_lines.append(f"{label} {p} ({_describe_bearing(along, perp)})")
        cached = "\n".join(cached_lines) if cached_lines else "No subtask target cached yet — explore to discover key/door/goal."

        target, allow_door = _current_target(env, state)
        if target is not None:
            hint = _bfs_hint_to_target(env, state, pos, env.agent_dir, target, allow_door=allow_door)
            if hint:
                hint_line = f"Planner hint (known cells, current subtask): {hint}\n"

        lw = _loop_warnings(state)
        state_block = f"""\
World position (x,y): {pos}
Has key: {has_key_str}  |  Door: {door_str}  |  Step: {env.step_count}/{env.max_steps}
{cached}
Visits at current cell: {state.visits.get(pos, 0)}
Visit counts (last up to 5 positions): {_visits_last_n(state, 5)}
Recent actions (L/R/F/P/D/T/E): {_format_recent_actions(state.actions, prompt_history)}
{lw}
{hint_line}"""

        if prompt_style == "stateful":
            map_block = f"""\
DISCOVERED MAP ({2 * prompt_map_radius + 1}×{2 * prompt_map_radius + 1} around you; A=you ?=unknown):
{render_known_window(state, env, prompt_map_radius)}

"""

    if rationale:
        rules = (
            "- Output only the two lines below (no markdown).\n"
            "- First line: Reason: <one short line, max 15 words>.\n"
            "- Second line: Action: <one of the VALID ACTIONS>.\n"
        )
        out_fmt = "Reason: <short>\nAction: TURN_LEFT or TURN_RIGHT or FORWARD or PICKUP or DROP or TOGGLE or DONE\n\n"
    else:
        rules = (
            "- Output only the action token on the Action: line (no markdown).\n"
            "- Do NOT explain.\n"
        )
        out_fmt = "Action: "

    rules += (
        "- PICKUP works only when the cell directly in front contains the key (k) and you are not carrying anything.\n"
        "- TOGGLE works only when the cell directly in front is a door (L/D/o); L requires the key in hand.\n"
        "- FORWARD: cannot move into '#' or into a closed/locked door — TOGGLE or turn first.\n"
        "- Use DONE only after the agent is on the goal G.\n"
    )
    if ppo_line:
        rules += "- If the autopilot suggestion is consistent with the current subtask and previews, follow it.\n"

    return f"""\
You are a navigation policy for a MiniGrid DoorKey task (partially observed).
Choose exactly ONE action.

VALID ACTIONS:
TURN_LEFT
TURN_RIGHT
FORWARD
PICKUP
DROP
TOGGLE
DONE

RULES:
{rules}
{legend}

CURRENT VIEW (7×7 egocentric):
{grid}

{map_block}STATUS:
Facing: {direction}
Has key: {has_key_str}
Door: {door_str}

CURRENT SUBTASK:
{subtask}{ppo_line}

{state_block}{enriched_block}
Examples:
key visible (2 forward, 1 right), no key in hand → TURN_RIGHT
facing 'k', no key in hand → PICKUP
key in hand, facing 'L' → TOGGLE
door open and goal 1 forward → FORWARD
visits high + LOOP_HINT, longest_corridor=right_in_view → TURN_RIGHT

OUTPUT FORMAT (MANDATORY):
{out_fmt}"""


def _summarize(logs: List[Dict], extra: Optional[Dict] = None) -> Dict:
    rewards = [l["reward"] for l in logs]
    successes = [l["success"] for l in logs]
    summary: Dict[str, Any] = {
        "n_episodes":   len(logs),
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_success": float(np.mean(successes)),
        "std_success":  float(np.std(successes)),
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
      main run      → "doorkey_sX_{model_tag}"
      ckpt ablation → "doorkey_sX_{model_tag}_{ckpt_tag}"
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
        {"trial": t.number, "threshold": t.params["threshold"],
         "reward": t.value, "state": str(t.state)}
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

def eval_ppo(model_path: str, size: int, n_episodes: int, seed_offset: int = 0):
    env = DoorKeyEnv(size=size)
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    model.policy.set_training_mode(False)

    logs = []
    for ep in tqdm(range(n_episodes), desc="PPO eval", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, steps, success = False, 0.0, 0, 0
        t0 = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)
            if terminated and float(reward) > 0:
                success = 1
            steps += 1
            done = terminated or truncated
        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward, "success": success,
            "steps": steps, "IR": 0.0, "OR": 0.0,
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
    size: int,
    n_episodes: int,
    seed_offset: int = 0,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
):
    env = DoorKeyEnv(size=size)
    slm = load_slm(slm_cfg)
    tag = short_model_name(slm_cfg["model"])
    decoding = decoding_for(prompt_rationale)

    logs = []
    for ep in tqdm(range(n_episodes), desc=f"SLM {tag}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, steps, success, invalid = False, 0.0, 0, 0, 0
        ep_state = new_episode_state()
        update_episode_state(ep_state, env, None)
        t0 = time.time()
        while not done:
            prompt = build_prompt(
                env, ep_state,
                prompt_style=prompt_style,
                rationale=prompt_rationale,
                prompt_history=prompt_history,
                prompt_map_radius=prompt_map_radius,
            )
            output = slm.generate(prompt, decoding)
            action = parse_action(output.text, rationale=prompt_rationale)
            if action is None:
                invalid += 1
                action = 2  # fallback: FORWARD
            obs, reward, terminated, truncated, _ = env.step(action)
            update_episode_state(ep_state, env, action)
            total_reward += float(reward)
            if terminated and float(reward) > 0:
                success = 1
            steps += 1
            done = terminated or truncated
        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward, "success": success,
            "steps": steps, "IR": 1.0, "OR": float("nan"),
            "slm_valid_rate": 1.0 - invalid / steps if steps else 0.0,
            "invalid_action_rate": invalid / steps if steps else 0.0,
            "episode_time_s": time.time() - t0,
        })
    env.close()
    del slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _summarize(logs, {"slm_model": slm_cfg["model"]}), logs


# =============================================================================
# ASK eval
# =============================================================================

def eval_ask(
    model: PPO,
    slm,
    size: int,
    threshold: float,
    n_episodes: int,
    seed_offset: int = 0,
    n_mc_samples: int = N_MC_SAMPLES,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
):
    env = DoorKeyEnv(size=size)
    logs = []
    decoding = decoding_for(prompt_rationale)

    for ep in tqdm(range(n_episodes), desc=f"ASK τ={threshold:.2f}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        done, total_reward, steps, success = False, 0.0, 0, 0
        slm_called, slm_valid, slm_overwrites, slm_invalid = 0, 0, 0, 0
        ep_state = new_episode_state()
        update_episode_state(ep_state, env, None)
        t0 = time.time()
        while not done:
            action_arr, _ = model.predict(obs, deterministic=True)
            ppo_action = int(action_arr)

            total_unc, _, _, _ = compute_mc_uncertainties(model, obs, n_samples=n_mc_samples)

            if total_unc >= threshold:
                slm_called += 1
                prompt = build_prompt(
                    env, ep_state, ppo_action,
                    prompt_style=prompt_style,
                    rationale=prompt_rationale,
                    prompt_history=prompt_history,
                    prompt_map_radius=prompt_map_radius,
                )
                output = slm.generate(prompt, decoding)
                slm_action = parse_action(output.text, rationale=prompt_rationale)
                if slm_action is not None:
                    slm_valid += 1
                    if slm_action != ppo_action:
                        ppo_action = slm_action
                        slm_overwrites += 1
                else:
                    slm_invalid += 1

            obs, reward, terminated, truncated, _ = env.step(ppo_action)
            update_episode_state(ep_state, env, ppo_action)
            total_reward += float(reward)
            if terminated and float(reward) > 0:
                success = 1
            steps += 1
            done = terminated or truncated

        logs.append({
            "episode": ep + 1, "seed": seed_offset + ep,
            "reward": total_reward, "success": success,
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
    size,
    n_eval_episodes,
    n_mc_samples,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
):
    threshold = trial.suggest_float("threshold", 0.01, 2.0)
    mean_reward, _ = eval_ask(
        model, slm, size, threshold, n_eval_episodes,
        seed_offset=0, n_mc_samples=n_mc_samples,
        prompt_style=prompt_style,
        prompt_rationale=prompt_rationale,
        prompt_history=prompt_history,
        prompt_map_radius=prompt_map_radius,
    )
    return mean_reward


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ppo", "slm", "ask"], default="ppo")
    p.add_argument("--size", type=int, default=5, choices=[5, 6, 8, 16])
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
    p.add_argument("--model-path", type=str, default=None, dest="model_path")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="doorkey", dest="wandb_group")
    p.add_argument(
        "--prompt-style",
        choices=list(_PROMPT_STYLES),
        default="basic",
        dest="prompt_style",
        help="basic | enriched | stateful_min | stateful",
    )
    p.add_argument(
        "--prompt-rationale",
        action="store_true",
        dest="prompt_rationale",
        help="ask the SLM to emit Reason:/Action: instead of an action token",
    )
    p.add_argument(
        "--prompt-history",
        type=int,
        default=8,
        dest="prompt_history",
        help="number of recent actions to include in the stateful prompt",
    )
    p.add_argument(
        "--prompt-map-radius",
        type=int,
        default=5,
        dest="prompt_map_radius",
        help="half-window of the discovered-map block in the stateful prompt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    torch.manual_seed(42)

    if args.model_path is None:
        args.model_path = f"runs/door_key/model_s{args.size}"

    file_tag = f"_s{args.size}" + (f"_{args.tag}" if args.tag else "")

    checkpoint_reward = None
    if args.tag.startswith("ckpt_r"):
        try:
            checkpoint_reward = int(args.tag[6:]) / 100.0
        except ValueError:
            pass

    if args.mode == "ppo":
        console.rule(f"[bold cyan]PPO — DoorKey-{args.size}x{args.size}[/bold cyan]")
        cfg_ppo = {"env": f"DoorKey-{args.size}x{args.size}", "n_episodes": args.n_episodes}
        if checkpoint_reward is not None:
            cfg_ppo["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"dk_eval_ppo{file_tag}",
                        group=args.wandb_group, job_type="eval_ppo", config=cfg_ppo):
            summary, logs = eval_ppo(args.model_path, args.size, args.n_episodes,
                                     seed_offset=TEST_SEEDS.start)
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
        console.rule(f"[bold cyan]SLM-only — {tag} — DoorKey-{args.size}x{args.size}[/bold cyan]")
        cfg_slm = {
            "env": f"DoorKey-{args.size}x{args.size}",
            "slm_model": model_name,
            "n_episodes": args.n_episodes,
            "prompt_style": args.prompt_style,
            "prompt_rationale": bool(args.prompt_rationale),
            "prompt_history": args.prompt_history,
            "prompt_map_radius": args.prompt_map_radius,
        }
        if checkpoint_reward is not None:
            cfg_slm["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"dk_eval_slm_{tag}{file_tag}",
                        group=args.wandb_group, job_type="eval_slm", config=cfg_slm):
            summary, logs = eval_slm_only(
                cfg, args.size, args.n_episodes,
                seed_offset=TEST_SEEDS.start,
                prompt_style=args.prompt_style,
                prompt_rationale=args.prompt_rationale,
                prompt_history=args.prompt_history,
                prompt_map_radius=args.prompt_map_radius,
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

        if args.threshold is not None:
            best_threshold = args.threshold
            console.rule(f"[bold cyan]ASK — {tag} τ={best_threshold:.4f} (fixed)[/bold cyan]")
            _opt_model = PPO.load(args.model_path,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
            _opt_slm = load_slm(cfg)
        else:
            console.rule(
                f"[bold cyan]ASK — {tag} Optuna ({args.n_optuna_trials} trials)[/bold cyan]"
            )
            study_name = f"dk_ask_s{args.size}_{tag}" + (f"_{args.tag}" if args.tag else "")
            study = optuna.create_study(
                direction="maximize", storage="sqlite:///optuna.db",
                study_name=study_name, load_if_exists=True,
            )
            _opt_model = PPO.load(args.model_path,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
            _opt_slm = load_slm(cfg)
            study.optimize(
                lambda t: objective(
                    t, _opt_model, _opt_slm, args.size,
                    args.n_eval_episodes, args.n_mc,
                    prompt_style=args.prompt_style,
                    prompt_rationale=args.prompt_rationale,
                    prompt_history=args.prompt_history,
                    prompt_map_radius=args.prompt_map_radius,
                ),
                n_trials=args.n_optuna_trials, show_progress_bar=True,
            )
            best_threshold = study.best_params["threshold"]
            console.print(Panel(
                f"Best τ = [green]{best_threshold:.4f}[/green]  |  "
                f"eval reward = [green]{study.best_value:.4f}[/green]",
                title="Optuna result", border_style="cyan",
            ))
            threshold_key = (f"doorkey_s{args.size}_{tag}"
                             + (f"_{args.tag}" if args.tag else ""))
            save_threshold(threshold_key, {
                "threshold":       best_threshold,
                "optuna_study":    study_name,
                "eval_reward":     study.best_value,
                "model_path":      args.model_path,
                "slm_model":       model_name,
                "n_trials":        args.n_optuna_trials,
                "n_mc_samples":    args.n_mc,
                "n_eval_episodes": args.n_eval_episodes,
                "env":             f"DoorKey-{args.size}x{args.size}",
                "saved_at":        datetime.now().isoformat(),
            })

        cfg_ask = {
            "env": f"DoorKey-{args.size}x{args.size}", "slm_model": model_name,
            "threshold": best_threshold, "n_mc_samples": args.n_mc,
            "n_episodes": args.n_episodes, "size": args.size,
            "prompt_style": args.prompt_style,
            "prompt_rationale": bool(args.prompt_rationale),
            "prompt_history": args.prompt_history,
            "prompt_map_radius": args.prompt_map_radius,
        }
        if checkpoint_reward is not None:
            cfg_ask["checkpoint_reward"] = checkpoint_reward
        with wandb.init(project=WANDB_PROJECT, name=f"dk_eval_ask_{tag}{file_tag}",
                        group=args.wandb_group, job_type="eval_ask", config=cfg_ask):
            _, logs = eval_ask(
                _opt_model, _opt_slm, args.size, best_threshold,
                args.n_episodes, seed_offset=TEST_SEEDS.start,
                n_mc_samples=args.n_mc,
                prompt_style=args.prompt_style,
                prompt_rationale=args.prompt_rationale,
                prompt_history=args.prompt_history,
                prompt_map_radius=args.prompt_map_radius,
            )
            del _opt_model, _opt_slm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            summary = _summarize(logs, {
                "slm_model": model_name, "threshold": best_threshold,
                "n_mc_samples": args.n_mc,
                "IR_mean": float(np.mean([l["IR"] for l in logs])),
                "OR_mean": float(np.mean([l["OR"] for l in logs])),
                "slm_valid_rate": float(np.mean([l["slm_valid_rate"] for l in logs])),
                "invalid_action_rate": float(
                    np.mean([l["invalid_action_rate"] for l in logs])
                ),
            })
            if checkpoint_reward is not None:
                summary["checkpoint_reward"] = checkpoint_reward
            _print_summary_table(f"ASK {tag} results", summary)
            wandb.run.summary.update(summary)
            save_summary(summary, RESULTS_DIR / f"ask_{tag}_results{file_tag}.json")
            save_csv(logs, RESULTS_DIR / f"ask_{tag}_episodes{file_tag}.csv")


if __name__ == "__main__":
    main()
