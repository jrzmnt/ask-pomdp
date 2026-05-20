"""POV image ↔ world mapping for SLM prompts (MiniGrid FourRooms)."""

from __future__ import annotations

import numpy as np

from ask.envs.fourrooms import FourRoomsEnv, OBJECT_TO_CHAR


def test_obs_cell_center_is_agent_pos():
    env = FourRoomsEnv()
    env.reset(seed=42)
    ap = env.agent_pos_abs
    hits = [(r, c) for r in range(7) for c in range(7) if env.obs_cell_to_world(r, c) == ap]
    assert len(hits) == 1
    r, c = hits[0]
    assert env.obs_cell_to_world(r, c) == ap
    env.close()


def test_front_world_cell_matches_image_projection():
    env = FourRoomsEnv()
    env.reset(seed=0)
    uw = env._env.unwrapped
    front = tuple(int(x) for x in (np.asarray(uw.agent_pos) + np.asarray(uw.dir_vec)))
    hits = [
        (r, c)
        for r in range(7)
        for c in range(7)
        if env.obs_cell_to_world(r, c) == front
    ]
    assert len(hits) >= 1
    env.close()


def test_goal_visible_projection_matches_grid():
    """If G appears in the 7×7 view, projected world coords match the grid goal."""
    env = FourRoomsEnv()
    for seed in range(500):
        env.reset(seed=seed)
        grid = env._env.unwrapped.grid
        raw = env.raw_obs
        if raw is None:
            continue
        img = raw["image"]
        for r in range(7):
            for c in range(7):
                ch = OBJECT_TO_CHAR.get(int(img[r, c, 0]), "?")
                if ch != "G":
                    continue
                wx, wy = env.obs_cell_to_world(r, c)
                cell = grid.get(wx, wy)
                assert cell is not None
                assert getattr(cell, "type", None) == "goal"
                env.close()
                return
    env.close()
    import pytest

    pytest.skip("Goal not in partial view for seeds 0..499")
