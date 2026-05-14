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
  python eval_ppo_slm.py --mode slm --slm qwen3.5-2b --prompt-style stateful --prompt-rationale
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
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

from ask.envs.fourrooms import DIR_TO_STR, FourRoomsEnv, OBJECT_TO_CHAR
from ask.slm.model import load_slm
from ask.uncertainty.entropy import compute_mc_uncertainties
from ask.utils.seed import set_seed

console = Console()


# =============================================================================
# Constants
# =============================================================================

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
DECODING_RATIONALE_MAX_TOKENS = 48

N_EVAL_EPISODES = 100
N_TEST_EPISODES = 100
N_MC_SAMPLES = 30

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ACTIONS_STR = ["TURN_LEFT", "TURN_RIGHT", "FORWARD"]
ACTION_LETTER = {0: "L", 1: "R", 2: "F"}

# MiniGrid agent_dir → unit forward vector (x, y)
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

_STR_TO_ACTION_ORDERED: List[Tuple[str, int]] = [
    ("TURN_LEFT", 0),
    ("TURN_RIGHT", 1),
    ("FORWARD", 2),
    ("LEFT", 0),
    ("RIGHT", 1),
    ("UP", 2),
]


# =============================================================================
# Prompt
# =============================================================================


def decoding_for(rationale: bool) -> Dict[str, Any]:
    d = dict(DECODING)
    if rationale:
        d["max_tokens"] = DECODING_RATIONALE_MAX_TOKENS
    return d


@dataclass
class EpisodeState:
    visits: Dict[Tuple[int, int], int] = field(default_factory=dict)
    known_grid: Dict[Tuple[int, int], str] = field(default_factory=dict)
    actions: List[int] = field(default_factory=list)
    goal_pos_abs: Optional[Tuple[int, int]] = None
    last_pos: Optional[Tuple[int, int]] = None
    stuck_steps: int = 0
    recent_pos_hist: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=5))


def new_episode_state() -> EpisodeState:
    return EpisodeState()


def _merge_known_cell(old: str, new: str) -> str:
    if new == "?":
        return old
    if new == "G":
        return "G"
    if new == "#":
        return "#"
    if new == "D":
        return "D" if old != "G" else "G"
    if old == "G":
        return "G"
    return new


def merge_obs_into_known_grid(state: EpisodeState, env: FourRoomsEnv) -> None:
    raw = env.raw_obs
    if raw is None:
        return
    img = raw["image"]
    for r in range(7):
        for c in range(7):
            ch = OBJECT_TO_CHAR.get(int(img[r, c, 0]), "?")
            if ch == "?":
                continue
            pos = env.obs_cell_to_world(r, c)
            prev = state.known_grid.get(pos, "?")
            state.known_grid[pos] = _merge_known_cell(prev, ch)
            if ch == "G":
                state.goal_pos_abs = pos


def update_episode_state(
    state: EpisodeState, env: FourRoomsEnv, action_taken: Optional[int]
) -> None:
    merge_obs_into_known_grid(state, env)
    pos = env.agent_pos_abs
    prev = state.last_pos
    if prev is not None and prev == pos:
        state.stuck_steps += 1
    else:
        state.stuck_steps = 0
    state.last_pos = pos
    state.visits[pos] = state.visits.get(pos, 0) + 1
    state.recent_pos_hist.append(pos)
    if action_taken is not None:
        state.actions.append(int(action_taken))
    if state.goal_pos_abs is None:
        for p, ch in state.known_grid.items():
            if ch == "G":
                state.goal_pos_abs = p
                break


def _world_cell_char(env: FourRoomsEnv, wx: int, wy: int) -> str:
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
        idx = OBJECT_TO_IDX.get(t, 1)
        return OBJECT_TO_CHAR.get(idx, ".")
    return "."


def _corridor_blocked(ch: str) -> bool:
    return ch in ("#", "?", "L")


def longest_corridor_world(env: FourRoomsEnv, pos: Tuple[int, int]) -> Tuple[str, int]:
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


def _agent_image_ij(env: FourRoomsEnv) -> Tuple[int, int]:
    ap = env.agent_pos_abs
    for r in range(7):
        for c in range(7):
            if env.obs_cell_to_world(r, c) == ap:
                return r, c
    return 3, 6


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


def _parse_grid_features(env: FourRoomsEnv) -> Dict[str, Any]:
    """Navigation features from the current POV and world geometry."""
    raw = env.raw_obs
    if raw is None:
        return {
            "ahead": "?",
            "left": "?",
            "right": "?",
            "back": "?",
            "ahead_chain": ["?", "?", "?"],
            "goal": None,
            "doors": [],
            "longest_corridor": ("forward_in_view", 0),
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

    goal_pos_img: Optional[Tuple[int, int]] = None
    for r in range(7):
        for c in range(7):
            ch = OBJECT_TO_CHAR.get(int(img[r, c, 0]), "?")
            if ch == "G":
                goal_pos_img = (r, c)
                break
        if goal_pos_img is not None:
            break

    ar, ac = _agent_image_ij(env)
    doors: List[str] = []
    for r in range(7):
        for c in range(7):
            ch = OBJECT_TO_CHAR.get(int(img[r, c, 0]), "?")
            if ch != "D":
                continue
            wx, wy = env.obs_cell_to_world(r, c)
            dw, dh = wx - ax, wy - ay
            doors.append(f"world_delta x={dw:+d} y={dh:+d} (image r={r} c={c})")

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
            ch = OBJECT_TO_CHAR.get(int(img[rr, cc, 0]), "?")
            if ch in ("#", "?", ""):
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
        "goal": goal_pos_img,
        "doors": doors,
        "longest_corridor": (best_name, best_len),
    }


def _goal_desc_ego(env: FourRoomsEnv, goal_img: Optional[Tuple[int, int]]) -> str:
    if not goal_img:
        return "not visible"
    gwx, gwy = env.obs_cell_to_world(goal_img[0], goal_img[1])
    along, perp = _world_delta_ego(env.agent_pos_abs, env.agent_dir, (gwx, gwy))
    if perp > 0:
        return f"visible ({along} forward, {perp} right in agent frame)"
    if perp < 0:
        return f"visible ({along} forward, {abs(perp)} left in agent frame)"
    return f"visible ({along} forward, straight)"


def _passable_ahead(ch: str) -> bool:
    return ch not in ("#",)


def _format_recent_actions(actions: List[int], n: int) -> str:
    tail = actions[-n:] if n > 0 else []
    return " ".join(ACTION_LETTER[a] for a in tail) if tail else "(none)"


def _visits_last_n(state: EpisodeState, n: int) -> str:
    seq = list(state.recent_pos_hist)[-n:]
    return str([state.visits.get(p, 0) for p in seq])


def _loop_warnings(state: EpisodeState) -> str:
    lines: List[str] = []
    if state.stuck_steps >= 6:
        lines.append(
            "LOOP_HINT: You have not moved to a new cell for several steps; "
            "prefer FORWARD if passable, otherwise turn toward a longer corridor."
        )
    if len(state.actions) >= 4 and all(a in (0, 1) for a in state.actions[-4:]):
        lines.append(
            "LOOP_HINT: Many consecutive turns; align with a clear corridor or cached goal bearing."
        )
    return "\n".join(lines)


def _room_id(x: int, y: int, w: int, h: int) -> str:
    mx, my = w // 2, h // 2
    ew = "W" if x < mx else "E"
    ns = "N" if y < my else "S"
    return ns + ew


def _bfs_hint_to_goal(
    _env: FourRoomsEnv,
    state: EpisodeState,
    agent_pos: Tuple[int, int],
    agent_dir: int,
) -> Optional[str]:
    goal = state.goal_pos_abs
    if goal is None or agent_pos == goal:
        return None
    known = state.known_grid
    passable = {p for p, ch in known.items() if ch not in ("#", "?")}
    if agent_pos not in passable or goal not in passable:
        return None
    from collections import deque as dq

    q: Deque[Tuple[int, int]] = dq([agent_pos])
    prev_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {agent_pos: None}
    reached = False
    while q:
        cur = q.popleft()
        if cur == goal:
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
    cur = goal
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
        return "FORWARD (one step on known map toward cached goal)"
    if (dx, dy) == (rdx, rdy):
        return "TURN_RIGHT then FORWARD when aligned (toward cached goal on known map)"
    if (dx, dy) == (-rdx, -rdy):
        return "TURN_LEFT then FORWARD when aligned (toward cached goal on known map)"
    if (dx, dy) == (-fdx, -fdy):
        return "TURN_LEFT twice (or RIGHT twice) to face cached goal, then FORWARD"
    return None


def _action_previews(env: FourRoomsEnv, pos: Tuple[int, int], agent_dir: int) -> List[str]:
    ax, ay = pos
    lines: List[str] = []
    # TURN_LEFT
    nd = (agent_dir - 1) % 4
    vx, vy = DIR_TO_VEC[nd]
    ach = _world_cell_char(env, ax + vx, ay + vy)
    lines.append(
        f"TURN_LEFT  → facing={DIR_TO_STR[nd]}, cell ahead would be {ach!r}"
    )
    nd = (agent_dir + 1) % 4
    vx, vy = DIR_TO_VEC[nd]
    ach = _world_cell_char(env, ax + vx, ay + vy)
    lines.append(
        f"TURN_RIGHT → facing={DIR_TO_STR[nd]}, cell ahead would be {ach!r}"
    )
    vx, vy = DIR_TO_VEC[agent_dir]
    nx, ny = ax + vx, ay + vy
    fch = _world_cell_char(env, nx, ny)
    if _corridor_blocked(fch):
        lines.append(f"FORWARD    → blocked (ahead is {fch!r})")
    else:
        lines.append(f"FORWARD    → move to ({nx},{ny}), ahead cell is {fch!r}")
    return lines


def render_known_window(
    state: EpisodeState,
    env: FourRoomsEnv,
    radius: int,
) -> str:
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


def build_prompt(
    env: FourRoomsEnv,
    state: Optional[EpisodeState] = None,
    ppo_action: Optional[int] = None,
    *,
    prompt_style: str = "basic",
    rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
) -> str:
    grid = env.render_view_ascii()
    direction = DIR_TO_STR.get(env.agent_dir, "UNKNOWN")
    features = _parse_grid_features(env)
    ahead = features["ahead"]
    ahead_passable = _passable_ahead(ahead)
    ahead_desc = "passable" if ahead_passable else "BLOCKED"
    goal_desc = _goal_desc_ego(env, features["goal"])
    ppo_line = (
        f"\nAutopilot suggests: {ACTIONS_STR[int(ppo_action)]}"
        if ppo_action is not None
        else ""
    )

    allowed_styles = ("basic", "enriched", "stateful_min", "stateful")
    if prompt_style not in allowed_styles:
        raise ValueError(f"prompt_style must be one of {allowed_styles}, got {prompt_style!r}")

    if prompt_style == "basic":
        rules_extra = (
            "- Do NOT explain.\n"
            "- Do NOT add text or markdown.\n"
        )
        out_fmt = "TURN_LEFT or TURN_RIGHT or FORWARD\n\nAction: "
        return f"""\
You are a robot navigation policy.
Your task is to choose exactly ONE action.

VALID ACTIONS:
TURN_LEFT
TURN_RIGHT
FORWARD

RULES:
{rules_extra}- If the path ahead is BLOCKED, do not choose FORWARD.
- If the goal is visible, prioritize moving toward it.
- If the autopilot suggestion is safe, follow it.

CURRENT VIEW (A=you  .=floor  #=wall  G=goal  ?=unseen  D=door):
{grid}

STATE:
Facing: {direction}
Path ahead: {ahead_desc}
Goal: {goal_desc}{ppo_line}

Examples:
ahead=passable, goal visible (3 ahead, straight) → FORWARD
ahead=passable, goal visible (2 ahead, 2 right) → TURN_RIGHT
ahead=passable, door visible (2 ahead, 2 right) → TURN_RIGHT
ahead=BLOCKED, door visible to the right → TURN_RIGHT
ahead=passable, goal not visible, longest corridor is right_in_view → TURN_RIGHT

OUTPUT FORMAT (MANDATORY):
{out_fmt}"""

    # ---- enriched / stateful* ----
    chain_s = " ".join(features["ahead_chain"])
    doors_s = ", ".join(features["doors"]) if features["doors"] else "(none visible)"
    lc_name, lc_len = features["longest_corridor"]
    pos = env.agent_pos_abs
    w, h = env._env.unwrapped.width, env._env.unwrapped.height
    lcw_name, lcw_len = longest_corridor_world(env, pos)
    previews = _action_previews(env, pos, env.agent_dir)

    enriched_block = f"""\
Adjacent (forward/left/right/back in view): {features['ahead']}/{features['left']}/{features['right']}/{features['back']}
Ahead chain (steps 1-3): {chain_s}
Doors visible (offsets): {doors_s}
Longest clear ray in view: {lc_name} ({lc_len} steps)
Longest clear ray in world frame: {lcw_name} ({lcw_len} steps)

ACTION PREVIEW (ground truth from simulator):
{chr(10).join(previews)}
"""

    state_block = ""
    map_block = ""
    hint_line = ""
    if prompt_style in ("stateful_min", "stateful") and state is not None:
        gx, gy = pos
        goal_cached = state.goal_pos_abs
        if goal_cached is not None:
            dgx, dgy = goal_cached[0] - gx, goal_cached[1] - gy
            mh = abs(dgx) + abs(dgy)
            goal_line = (
                f"Goal cached at world {goal_cached}  "
                f"(delta x={dgx:+d}, y={dgy:+d}, manhattan={mh})"
            )
            hint = _bfs_hint_to_goal(env, state, pos, env.agent_dir)
            if hint:
                hint_line = f"Planner hint (known cells): {hint}\n"
        else:
            goal_line = "Goal not yet seen — explore through doors and along long corridors."
        room = _room_id(gx, gy, w, h)
        gr_line = ""
        if goal_cached is not None:
            gr = _room_id(goal_cached[0], goal_cached[1], w, h)
            gr_line = f"Goal room (cached): {gr}\n"
        lw = _loop_warnings(state)
        state_block = f"""\
World position (x,y): {pos}
Room (quadrant): {room}
{gr_line}{goal_line}
Visits at current cell: {state.visits.get(pos, 0)}
Visit counts (last up to 5 positions): {_visits_last_n(state, 5)}
Recent actions (L/R/F): {_format_recent_actions(state.actions, prompt_history)}
{lw}
{hint_line}"""
        if prompt_style == "stateful":
            map_block = f"""\
DISCOVERED MAP ({2 * prompt_map_radius + 1}×{2 * prompt_map_radius + 1} around you; A=you ?=unknown):
{render_known_window(state, env, prompt_map_radius)}

"""

    rules_stateful = (
        "- Output only the action token line required below (no markdown).\n"
    )
    if rationale:
        rules_stateful += (
            "- First line: Reason: <one short line, max 15 words>.\n"
            "- Second line: Action: <TURN_LEFT|TURN_RIGHT|FORWARD>.\n"
        )
    else:
        rules_stateful += "- Do NOT explain.\n"

    out_fmt2 = (
        "Reason: <short>\nAction: TURN_LEFT or TURN_RIGHT or FORWARD\n\n"
        if rationale
        else "TURN_LEFT or TURN_RIGHT or FORWARD\n\nAction: "
    )

    return f"""\
You are a robot navigation policy in MiniGrid FourRooms (partially observed).
Choose exactly ONE low-level action.

VALID ACTIONS:
TURN_LEFT
TURN_RIGHT
FORWARD

RULES:
{rules_stateful}- If FORWARD is blocked (ahead is '#' or you cannot enter), do not choose FORWARD.
- Prefer moving through doors (D) when they align with corridors or the cached goal.
- If the egocentric goal 'G' is visible, move toward it.
- If a planner hint is given on the known map, prefer it when it matches safe previews.
{('- If autopilot suggests a safe action, follow it.' if ppo_line else '')}

CURRENT VIEW (A=you  .=floor  #=wall  G=goal  ?=unseen  D=door):
{grid}

{map_block}STATE:
Facing: {direction}
Path ahead: {ahead_desc}
Goal (in view): {goal_desc}
{state_block}{enriched_block}{ppo_line}

Examples:
ahead=passable, goal visible (3 ahead, straight) → FORWARD
ahead=passable, goal visible (2 ahead, 2 right) → TURN_RIGHT
ahead=passable, door visible (2 ahead, 2 right) → TURN_RIGHT
ahead=BLOCKED, door visible to the right → TURN_RIGHT
visits high + loop hint + longest corridor is right_in_view → TURN_RIGHT

OUTPUT FORMAT (MANDATORY):
{out_fmt2}"""


# =============================================================================
# Action parsing
# =============================================================================


def _parse_action_substring(text_upper: str) -> Optional[int]:
    for key, val in _STR_TO_ACTION_ORDERED:
        if key in text_upper:
            return val
    return None


def parse_action(text: str, *, rationale: bool = False) -> Optional[int]:
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
# Helpers
# =============================================================================

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def short_model_name(model_name: str) -> str:
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
            "episode":             ep + 1,
            "seed":                seed_offset + ep,
            "reward":              ep_reward,
            "length":              ep_len,
            "result":              "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "IR":                  0.0,
            "OR":                  0.0,
            "slm_valid_rate":      0.0,
            "invalid_action_rate": 0.0,
            "episode_time_s":      time.time() - t0,
        })

    env.close()
    return _summarize(logs), logs


# =============================================================================
# SLM-only evaluation
# =============================================================================

def eval_slm_only(
    slm_cfg: Dict[str, Any],
    n_episodes: int,
    seed_offset: int = 0,
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = FourRoomsEnv()
    slm = load_slm(slm_cfg)

    decode = decoding_for(prompt_rationale)
    model_tag = short_model_name(slm_cfg["model"])
    logs = []
    for ep in tqdm(range(n_episodes), desc=f"SLM {model_tag}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_state: Optional[EpisodeState] = None
        if prompt_style in ("stateful_min", "stateful"):
            ep_state = new_episode_state()
            update_episode_state(ep_state, env, None)
        done, ep_reward, ep_len = False, 0.0, 0
        invalid_actions = 0
        t0 = time.time()
        while not done:
            prompt = build_prompt(
                env,
                ep_state,
                ppo_action=None,
                prompt_style=prompt_style,
                rationale=prompt_rationale,
                prompt_history=prompt_history,
                prompt_map_radius=prompt_map_radius,
            )
            output = slm.generate(prompt, decode)
            action = parse_action(output.text, rationale=prompt_rationale)
            if action is None:
                invalid_actions += 1
                action = 2  # fallback: FORWARD
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
            if ep_state is not None:
                update_episode_state(ep_state, env, int(action))
        ep_invalid_rate = invalid_actions / ep_len if ep_len > 0 else 0.0

        data = {
            "episode":             ep + 1,
            "seed":                seed_offset + ep,
            "reward":              ep_reward,
            "length":              ep_len,
            "result":              "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "IR":                  1.0,
            "OR":                  float("nan"),  # no PPO reference in SLM-only
            "slm_valid_rate":      1.0 - ep_invalid_rate,
            "invalid_action_rate": ep_invalid_rate,
            "episode_time_s":      time.time() - t0,
        }
        print(data)
        logs.append(data)

    env.close()
    del slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extra = {
        "slm_model":           slm_cfg["model"],
        "invalid_action_rate": float(np.mean([l["invalid_action_rate"] for l in logs])),
        "prompt_style":        prompt_style,
        "prompt_rationale":    prompt_rationale,
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
    *,
    prompt_style: str = "basic",
    prompt_rationale: bool = False,
    prompt_history: int = 8,
    prompt_map_radius: int = 5,
) -> tuple[float, List[Dict[str, Any]]]:
    env = FourRoomsEnv()
    logs: List[Dict[str, Any]] = []
    decode = decoding_for(prompt_rationale)

    for ep in tqdm(range(n_episodes), desc=f"ASK τ={threshold:.2f}", unit="ep", leave=False):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_state: Optional[EpisodeState] = None
        if prompt_style in ("stateful_min", "stateful"):
            ep_state = new_episode_state()
            update_episode_state(ep_state, env, None)
        done, ep_reward, steps = False, 0.0, 0
        slm_called, slm_valid, slm_overwrites, slm_invalid = 0, 0, 0, 0

        t0 = time.time()
        while not done:
            action_arr, _ = model.predict(obs, deterministic=True)
            ppo_action = int(action_arr)

            total_unc, _, _, _ = compute_mc_uncertainties(model, obs, n_samples=n_mc_samples)

            if total_unc >= threshold:
                slm_called += 1
                prompt = build_prompt(
                    env,
                    ep_state,
                    ppo_action,
                    prompt_style=prompt_style,
                    rationale=prompt_rationale,
                    prompt_history=prompt_history,
                    prompt_map_radius=prompt_map_radius,
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

            obs, reward, terminated, truncated, _ = env.step(ppo_action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            if ep_state is not None:
                update_episode_state(ep_state, env, int(ppo_action))

        logs.append({
            "episode":             ep + 1,
            "seed":                seed_offset + ep,
            "reward":              ep_reward,
            "length":              steps,
            "result":              "goal" if ep_reward > 0 else "timeout" if not terminated else "failure",
            "IR":                  slm_called / steps if steps > 0 else 0.0,
            "OR":                  slm_overwrites / steps if steps > 0 else 0.0,
            "slm_valid_rate":      slm_valid / slm_called if slm_called > 0 else 0.0,
            "invalid_action_rate": slm_invalid / slm_called if slm_called > 0 else 0.0,
            "episode_time_s":      time.time() - t0,
        })
        print(logs[-1])

    env.close()
    return float(np.mean([l["reward"] for l in logs])), logs


# =============================================================================
# Optuna objective
# =============================================================================

def objective(
    trial: optuna.Trial,
    model: "PPO",
    slm: "HuggingFaceSLM",
    n_eval_episodes: int,
    n_mc_samples: int,
    *,
    prompt_style: str,
    prompt_rationale: bool,
    prompt_history: int,
    prompt_map_radius: int,
) -> float:
    threshold = trial.suggest_float("threshold", 0.1, 2.0)
    mean_reward, _ = eval_ask(
        model=model, slm=slm, threshold=threshold,
        n_episodes=n_eval_episodes, seed_offset=0, n_mc_samples=n_mc_samples,
        prompt_style=prompt_style,
        prompt_rationale=prompt_rationale,
        prompt_history=prompt_history,
        prompt_map_radius=prompt_map_radius,
    )
    if mean_reward >= 0.999:
        trial.study.stop()
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


def save_threshold(key: str, entry: Dict[str, Any]) -> None:
    """Persist Optuna threshold under a structured key.

    Key format:
      main run       → "fourrooms_{model_tag}"
      ckpt ablation  → "fourrooms_{model_tag}_{ckpt_tag}"
    """
    path = RESULTS_DIR / "thresholds.json"
    registry: Dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            registry = json.load(f)
    registry[key] = entry
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    console.print(f"  Threshold saved → {path} [{key}]")


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
    p.add_argument("--slm", choices=list(QWEN_MODELS.keys()) + ["all"], default="all")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed τ — skips Optuna")
    p.add_argument("--n-mc", type=int, default=N_MC_SAMPLES, dest="n_mc")
    p.add_argument("--n-episodes", type=int, default=N_TEST_EPISODES, dest="n_episodes")
    p.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES, dest="n_eval_episodes")
    p.add_argument("--n-optuna-trials", type=int, default=15, dest="n_optuna_trials")
    p.add_argument("--model-path", type=str, default=None, dest="model_path")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix for output files, e.g. 'mc10'")
    p.add_argument("--wandb-group", type=str, default="fourrooms", dest="wandb_group",
                   help="W&B run group")
    p.add_argument(
        "--prompt-style",
        choices=["basic", "enriched", "stateful_min", "stateful"],
        default="basic",
        dest="prompt_style",
        help="SLM prompt: basic (legacy), enriched (+local cues), stateful_min (+memory), stateful (+map)",
    )
    p.add_argument(
        "--prompt-rationale",
        action="store_true",
        dest="prompt_rationale",
        help="Allow one-line Reason: before Action: (increases max_new_tokens)",
    )
    p.add_argument("--prompt-history", type=int, default=8, dest="prompt_history",
                   help="Recent actions shown in stateful prompts")
    p.add_argument("--prompt-map-radius", type=int, default=5, dest="prompt_map_radius",
                   help="Half-size of discovered-map window (stateful)")
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
    slm_keys = ["qwen3.5-2b", "qwen3.5-4b"] if args.slm == "all" else [args.slm]

    # Extract checkpoint reward from tag (e.g. "ckpt_r030" → 0.30)
    checkpoint_reward = None
    if args.tag.startswith("ckpt_r"):
        try:
            checkpoint_reward = int(args.tag[6:]) / 100.0
        except ValueError:
            pass

    is_ablation = bool(args.tag)
    group = args.wandb_group
    job_type_ask = "ablation" if is_ablation else "eval_ask"

    # -------------------------------------------------------------------------
    # PPO-only baseline
    # -------------------------------------------------------------------------
    if args.mode in ("ppo", "all"):
        console.rule("[bold cyan]PPO baseline[/bold cyan]")
        cfg_ppo = {
            "env": "MiniGrid-FourRooms-v0",
            "n_episodes": args.n_episodes,
        }
        if checkpoint_reward is not None:
            cfg_ppo["checkpoint_reward"] = checkpoint_reward
        with wandb.init(
            project=WANDB_PROJECT,
            name=f"eval_ppo{file_tag}",
            group=group,
            job_type="eval_ppo",
            config=cfg_ppo,
        ):
            summary, logs = eval_ppo(model_path, n_episodes=args.n_episodes, seed_offset=N_EVAL_EPISODES)
            _print_summary_table("PPO results", summary)
            if checkpoint_reward is not None:
                summary["checkpoint_reward"] = checkpoint_reward
            wandb_log_summary(wandb.run, summary)
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
            cfg_slm = {
                "env": "MiniGrid-FourRooms-v0",
                "slm_model": model_name,
                "n_episodes": args.n_episodes,
                "prompt_style": args.prompt_style,
                "prompt_rationale": args.prompt_rationale,
                "prompt_history": args.prompt_history,
                "prompt_map_radius": args.prompt_map_radius,
            }
            if checkpoint_reward is not None:
                cfg_slm["checkpoint_reward"] = checkpoint_reward
            with wandb.init(
                project=WANDB_PROJECT,
                name=f"eval_slm_{tag}{file_tag}",
                group=group,
                job_type="eval_slm",
                config=cfg_slm,
            ):
                summary, logs = eval_slm_only(
                    cfg,
                    n_episodes=args.n_episodes,
                    seed_offset=N_EVAL_EPISODES,
                    prompt_style=args.prompt_style,
                    prompt_rationale=args.prompt_rationale,
                    prompt_history=args.prompt_history,
                    prompt_map_radius=args.prompt_map_radius,
                )
                _print_summary_table(f"SLM {tag} results", summary)
                wandb_log_summary(wandb.run, summary)
                save_summary(summary, RESULTS_DIR / f"slm_{tag}_results{file_tag}.json")
                save_csv(logs, RESULTS_DIR / f"slm_{tag}_episodes{file_tag}.csv")

        # --- ASK ---
        if args.mode in ("ask", "all"):
            wandb_cfg = {
                "env": "MiniGrid-FourRooms-v0",
                "slm_model": model_name,
                "n_mc_samples": args.n_mc,
                "n_episodes": args.n_episodes,
                "n_eval_episodes": args.n_eval_episodes,
                "prompt_style": args.prompt_style,
                "prompt_rationale": args.prompt_rationale,
                "prompt_history": args.prompt_history,
                "prompt_map_radius": args.prompt_map_radius,
            }
            if checkpoint_reward is not None:
                wandb_cfg["checkpoint_reward"] = checkpoint_reward

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
                # Load model + SLM once — reused across all trials and final eval
                _opt_model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
                _opt_slm = load_slm(cfg)
                study.optimize(
                    lambda t: objective(
                        t, _opt_model, _opt_slm, args.n_eval_episodes, args.n_mc,
                        prompt_style=args.prompt_style,
                        prompt_rationale=args.prompt_rationale,
                        prompt_history=args.prompt_history,
                        prompt_map_radius=args.prompt_map_radius,
                    ),
                    n_trials=args.n_optuna_trials,
                    show_progress_bar=True,
                )
                best_threshold = study.best_params["threshold"]
                console.print(
                    Panel(
                        f"Best τ = [green]{best_threshold:.4f}[/green]  |  "
                        f"eval reward = [green]{study.best_value:.4f}[/green]  |  "
                        f"study = [cyan]{study_name}[/cyan]",
                        title="Optuna result", border_style="cyan",
                    )
                )
                wandb_cfg["threshold"] = best_threshold
                wandb_cfg["threshold_source"] = "optuna"
                wandb_cfg["optuna_study_name"] = study_name
                wandb_cfg["optuna_best_reward"] = study.best_value
                wandb_cfg["optuna_n_trials"] = args.n_optuna_trials
                threshold_key = f"fourrooms_{tag}" + (f"_{args.tag}" if args.tag else "")
                save_threshold(threshold_key, {
                    "threshold":       best_threshold,
                    "optuna_study":    study_name,
                    "eval_reward":     study.best_value,
                    "model_path":      str(model_path),
                    "slm_model":       model_name,
                    "n_trials":        args.n_optuna_trials,
                    "n_mc_samples":    args.n_mc,
                    "n_eval_episodes": args.n_eval_episodes,
                    "env":             "MiniGrid-FourRooms-v0",
                    "saved_at":        datetime.now().isoformat(),
                })

            with wandb.init(
                project=WANDB_PROJECT,
                name=f"eval_ask_{tag}{file_tag}",
                group=group,
                job_type=job_type_ask,
                config=wandb_cfg,
            ):
                if args.threshold is not None:
                    # Fixed threshold path: load fresh
                    _opt_model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
                    _opt_slm = load_slm(cfg)

                _, logs = eval_ask(
                    model=_opt_model, slm=_opt_slm, threshold=best_threshold,
                    n_episodes=args.n_episodes, seed_offset=args.n_eval_episodes,
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

                summary = _summarize(logs, extra={
                    "slm_model":          model_name,
                    "threshold":          best_threshold,
                    "n_mc_samples":       args.n_mc,
                    "IR_mean":            float(np.mean([l["IR"] for l in logs])),
                    "IR_std":             float(np.std([l["IR"] for l in logs])),
                    "OR_mean":            float(np.mean([l["OR"] for l in logs])),
                    "OR_std":             float(np.std([l["OR"] for l in logs])),
                    "slm_valid_rate":     float(np.mean([l["slm_valid_rate"] for l in logs])),
                    "invalid_action_rate": float(np.mean([l["invalid_action_rate"] for l in logs])),
                })
                if checkpoint_reward is not None:
                    summary["checkpoint_reward"] = checkpoint_reward
                _print_summary_table(f"ASK {tag} results", summary)
                wandb_log_summary(wandb.run, summary)
                save_summary(summary, RESULTS_DIR / f"ask_{tag}_results{file_tag}.json")
                save_csv(logs, RESULTS_DIR / f"ask_{tag}_episodes{file_tag}.csv")


if __name__ == "__main__":
    main()
