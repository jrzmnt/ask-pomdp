from __future__ import annotations

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid environments

# MiniGrid object type → ASCII character for SLM prompts
OBJECT_TO_CHAR = {
    0: "?",   # unseen
    1: ".",   # empty
    2: "#",   # wall
    3: ".",   # floor
    4: "D",   # door (open or closed)
    5: "k",   # key
    6: "b",   # ball
    7: "B",   # box
    8: "G",   # goal
    9: "L",   # lava
    10: "A",  # agent
}

DIR_TO_STR = {0: "EAST", 1: "SOUTH", 2: "WEST", 3: "NORTH"}

# MiniGrid actions used for navigation
ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_FORWARD = 2


class FourRoomsEnv(gym.Env):
    """
    MiniGrid-FourRooms-v0 wrapped for PPO training.

    Observation: (147,) float32 — flattened 7×7×3 egocentric view, normalized to [0, 1].
    Action space: Discrete(3) — TURN_LEFT, TURN_RIGHT, FORWARD.

    The full raw observation dict and agent direction are accessible via
    `raw_obs` and `agent_dir` for SLM prompt construction.
    """

    def __init__(self, max_steps: int = 500):
        self._env = gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps)

        # Flat observation: 7×7 grid, 3 channels, normalized
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(7 * 7 * 3,), dtype=np.float32
        )
        # Restrict to the 3 navigation actions
        self.action_space = gym.spaces.Discrete(3)

        self._last_raw: dict | None = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        raw, info = self._env.reset(**kwargs)
        self._last_raw = raw
        return self._flatten(raw), info

    def step(self, action: int):
        raw, reward, terminated, truncated, info = self._env.step(action)
        self._last_raw = raw
        return self._flatten(raw), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # SLM helpers
    # ------------------------------------------------------------------

    @property
    def raw_obs(self) -> dict | None:
        return self._last_raw

    @property
    def agent_dir(self) -> int:
        return int(self._env.unwrapped.agent_dir)

    def render_view_ascii(self) -> str:
        """Return the current 7×7 egocentric view as an ASCII string."""
        if self._last_raw is None:
            return ""
        image = self._last_raw["image"]  # (7, 7, 3) uint8
        lines = []
        for row in image:
            lines.append("".join(OBJECT_TO_CHAR.get(int(cell[0]), "?") for cell in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(raw_obs: dict) -> np.ndarray:
        return (raw_obs["image"].astype(np.float32) / 10.0).flatten()
