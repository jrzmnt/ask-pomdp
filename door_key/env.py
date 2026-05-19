"""
DoorKey environment wrapper with subtask-aware state for prompt building.

Observation: 7×7×3 uint8 MiniGrid egocentric view → flattened (147,) float32.
Actions: TURN_LEFT(0), TURN_RIGHT(1), FORWARD(2), PICKUP(3), DROP(4), TOGGLE(5), DONE(6)
Reward: 1 - 0.9*(steps/max_steps) on success, 0 on failure.

Sequential subtasks: find key → PICKUP → find door → TOGGLE (unlock) → reach goal.

Seed protocol (stable methodology):
  TRAIN_SEEDS = 200–999   (800 maps; never used in val/test)
  VAL_SEEDS   = 0–99      (Optuna τ search)
  TEST_SEEDS  = 100–199   (final reported results)

Use SeededDoorKeyEnv to enforce these ranges automatically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid environments
import numpy as np

SIZES = {5: 250, 6: 360, 8: 640, 16: 2560}

# Seed ranges — keep disjoint to prevent map leakage across splits.
TRAIN_SEEDS = range(200, 1000)   # 800 maps
VAL_SEEDS   = range(0,   100)    # 100 maps — Optuna tau search
TEST_SEEDS  = range(100, 200)    # 100 maps — final reported results

ACTIONS = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]
_STR_TO_ACTION: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
_STR_TO_ACTION.update({"LEFT": 0, "RIGHT": 1, "FWD": 2, "PICK": 3, "PICK_UP": 3})

DIR_TO_STR = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

OBJECT_TO_CHAR = {
    0: " ",   # unseen
    1: ".",   # empty
    2: "#",   # wall
    3: ".",   # floor
    4: "D",   # door (refined by state channel below)
    5: "k",   # key
    6: "b",   # ball
    7: "B",   # box
    8: "G",   # goal
    9: "~",   # lava
    10: "A",  # agent
}

DOOR_STATE_CHAR = {0: "o", 1: "D", 2: "L"}  # open, closed, locked


class DoorKeyEnv(gym.Env):
    """
    MiniGrid-DoorKey wrapped for PPO training with subtask-aware prompt building.

    Supports grid sizes 5, 6, 8, 16. The wrapper tracks has_key / door_open
    state for subtask-aware SLM prompts.
    """

    def __init__(self, size: int = 5):
        assert size in SIZES, f"size must be one of {list(SIZES.keys())}"
        self.size = size
        self._max_steps = SIZES[size]
        self._env = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0", max_steps=self._max_steps)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(7 * 7 * 3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(7)

        self._last_raw: Optional[Dict[str, Any]] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        raw, info = self._env.reset(seed=seed, **kwargs)
        self._last_raw = raw
        self._step_count = 0
        return self._flatten(raw), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        raw, reward, terminated, truncated, info = self._env.step(action)
        self._last_raw = raw
        self._step_count += 1
        return self._flatten(raw), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def has_key(self) -> bool:
        """True if the agent is currently carrying the key."""
        return self._env.unwrapped.carrying is not None

    @property
    def door_open(self) -> bool:
        """True if the room's door is open."""
        for obj in self._env.unwrapped.grid.grid:
            if obj is not None and obj.type == "door":
                return obj.is_open
        return False

    @property
    def agent_dir(self) -> int:
        return int(self._env.unwrapped.agent_dir)

    # ------------------------------------------------------------------
    # SLM helpers
    # ------------------------------------------------------------------

    def render_view_ascii(self) -> str:
        """Return the current 7×7 egocentric view as a 7-line ASCII string."""
        if self._last_raw is None:
            return ""
        image = self._last_raw["image"]  # (7, 7, 3) uint8
        lines = []
        for row in image:
            chars = []
            for cell in row:
                obj_type = int(cell[0])
                state = int(cell[2])
                if obj_type == 4:  # door — use state channel
                    chars.append(DOOR_STATE_CHAR.get(state, "D"))
                else:
                    chars.append(OBJECT_TO_CHAR.get(obj_type, "?"))
            lines.append("".join(chars))
        return "\n".join(lines)

    def _current_subtask(self) -> str:
        if self.door_open:
            return "REACH THE GREEN GOAL TILE (G)"
        if self.has_key:
            return "FIND AND UNLOCK THE DOOR (face door, then TOGGLE)"
        return "FIND AND PICK UP THE KEY (face key, then PICKUP)"

    def build_prompt(self, ppo_action: Optional[int] = None) -> str:
        grid = self.render_view_ascii()
        direction = DIR_TO_STR.get(self.agent_dir, "?")
        subtask = self._current_subtask()
        has_key_str = "YES" if self.has_key else "NO"
        door_str = "OPEN" if self.door_open else "LOCKED"
        ppo_line = f"\nAutopilot suggests: {ACTIONS[ppo_action]}" if ppo_action is not None else ""

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
- Do NOT explain.
- Do NOT add text or markdown.
- PICKUP: pick up the key when you face it (adjacent, facing it).
- TOGGLE: unlock/open the door when you face it (requires key in hand).
- FORWARD: move one step in the direction you face.
- Avoid walls (#). Navigate around them.

LEGEND: A=agent, k=key, D=door(closed), L=door(locked), o=door(open), G=goal, #=wall, .=empty

CURRENT VIEW (7×7 egocentric, agent faces right of grid):
{grid}

STATUS:
Facing: {direction}
Has key: {has_key_str}
Door: {door_str}
Step: {self._step_count}/{self._max_steps}

CURRENT SUBTASK:
{subtask}{ppo_line}

Action: """

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(raw_obs: Dict) -> np.ndarray:
        return (raw_obs["image"].astype(np.float32) / 10.0).flatten()


class SeededDoorKeyEnv(DoorKeyEnv):
    """DoorKeyEnv that cycles through a fixed seed range when reset without an explicit seed.

    Use this to enforce train/val/test map splits:
      SeededDoorKeyEnv(size=5, seeds=TRAIN_SEEDS)  # training
      SeededDoorKeyEnv(size=5, seeds=VAL_SEEDS)    # eval callback during training
      # Final eval uses DoorKeyEnv + explicit seeds in eval.py loops
    """

    def __init__(self, size: int = 5, seeds: range = TRAIN_SEEDS):
        super().__init__(size=size)
        self._seed_pool = list(seeds)
        self._seed_idx = 0

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        if seed is None:
            seed = self._seed_pool[self._seed_idx % len(self._seed_pool)]
            self._seed_idx += 1
        return super().reset(seed=seed, **kwargs)
