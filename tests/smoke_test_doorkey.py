"""
Smoke tests for the MiniGrid DoorKey pipeline.

Strategy
--------
Unit tests (render_view_ascii, build_prompt, _current_subtask, parse_action) run on
synthetic numpy arrays using a duck-typed _FakeEnv — NO real gymnasium env needed.

Integration tests (env.reset/step, build_prompt on real obs, MC uncertainty) are
skipped unless gymnasium + minigrid are both importable.

Runtime (unit tests only): < 2 s.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from door_key.env import (
    ACTIONS,
    DIR_TO_STR,
    DOOR_STATE_CHAR,
    OBJECT_TO_CHAR,
    DoorKeyEnv,
    _STR_TO_ACTION,
)
from door_key.eval import parse_action


# ---------------------------------------------------------------------------
# Synthetic image factory: 7×7×3 uint8
# ---------------------------------------------------------------------------

def _make_image(
    agent_pos: tuple = (3, 3),
    key_pos: tuple | None = None,
    door_pos: tuple | None = None,
    door_state: int = 2,   # 0=open, 1=closed, 2=locked
    goal_pos: tuple | None = (6, 3),
) -> np.ndarray:
    image = np.ones((7, 7, 3), dtype=np.uint8)  # default: empty type=1
    image[:, :, 0] = 1
    image[:, :, 1] = 1
    image[:, :, 2] = 0

    # Border walls
    image[0, :, 0] = 2
    image[6, :, 0] = 2
    image[:, 0, 0] = 2
    image[:, 6, 0] = 2

    if agent_pos:
        image[agent_pos[0], agent_pos[1], 0] = 10  # agent
    if key_pos:
        image[key_pos[0], key_pos[1], 0] = 5       # key
    if door_pos:
        image[door_pos[0], door_pos[1], 0] = 4     # door
        image[door_pos[0], door_pos[1], 2] = door_state
    if goal_pos:
        image[goal_pos[0], goal_pos[1], 0] = 8     # goal

    return image


def _make_fake_env(
    image: np.ndarray | None = None,
    has_key: bool = False,
    door_open: bool = False,
    agent_dir: int = 0,
    step: int = 0,
    size: int = 5,
) -> DoorKeyEnv:
    """Duck-typed DoorKeyEnv for unit tests — no real gymnasium env needed."""
    if image is None:
        image = _make_image()

    fake = object.__new__(DoorKeyEnv)
    fake._last_raw = {"image": image}
    fake._step_count = step
    fake._max_steps = {5: 250, 6: 360, 8: 640, 16: 2560}[size]
    fake.size = size

    door_mock = MagicMock()
    door_mock.type = "door"
    door_mock.is_open = door_open
    door_mock.is_locked = not door_open

    unwrapped = MagicMock()
    unwrapped.carrying = MagicMock() if has_key else None
    unwrapped.grid.grid = [door_mock]
    unwrapped.agent_dir = agent_dir

    env_inner = MagicMock()
    env_inner.unwrapped = unwrapped
    fake._env = env_inner

    return fake


# ---------------------------------------------------------------------------
# Unit tests: render_view_ascii
# ---------------------------------------------------------------------------

def test_render_ascii_shape():
    lines = _make_fake_env().render_view_ascii().split("\n")
    assert len(lines) == 7
    assert all(len(l) == 7 for l in lines)


def test_render_ascii_agent():
    env = _make_fake_env(image=_make_image(agent_pos=(3, 3)))
    assert env.render_view_ascii().split("\n")[3][3] == "A"


def test_render_ascii_key():
    env = _make_fake_env(image=_make_image(key_pos=(2, 4)))
    assert env.render_view_ascii().split("\n")[2][4] == "k"


def test_render_ascii_goal():
    env = _make_fake_env(image=_make_image(goal_pos=(5, 3)))
    assert env.render_view_ascii().split("\n")[5][3] == "G"


def test_render_ascii_locked_door():
    env = _make_fake_env(image=_make_image(door_pos=(3, 5), door_state=2))
    assert env.render_view_ascii().split("\n")[3][5] == "L"


def test_render_ascii_closed_door():
    env = _make_fake_env(image=_make_image(door_pos=(3, 5), door_state=1))
    assert env.render_view_ascii().split("\n")[3][5] == "D"


def test_render_ascii_open_door():
    env = _make_fake_env(image=_make_image(door_pos=(3, 5), door_state=0))
    assert env.render_view_ascii().split("\n")[3][5] == "o"


def test_render_ascii_wall_borders():
    lines = _make_fake_env().render_view_ascii().split("\n")
    assert all(c == "#" for c in lines[0])   # top border
    assert lines[1][0] == "#"                # left border


def test_render_ascii_empty_when_no_obs():
    fake = object.__new__(DoorKeyEnv)
    fake._last_raw = None
    assert fake.render_view_ascii() == ""


# ---------------------------------------------------------------------------
# Unit tests: _current_subtask
# ---------------------------------------------------------------------------

def test_subtask_find_key():
    env = _make_fake_env(has_key=False, door_open=False)
    subtask = env._current_subtask()
    assert "KEY" in subtask.upper()


def test_subtask_unlock_door():
    env = _make_fake_env(has_key=True, door_open=False)
    subtask = env._current_subtask()
    assert "DOOR" in subtask.upper() or "TOGGLE" in subtask.upper()


def test_subtask_reach_goal():
    env = _make_fake_env(has_key=False, door_open=True)
    subtask = env._current_subtask()
    assert "GOAL" in subtask.upper()


# ---------------------------------------------------------------------------
# Unit tests: build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_agent():
    assert "A" in _make_fake_env().build_prompt()


def test_build_prompt_all_actions_present():
    prompt = _make_fake_env().build_prompt()
    for action in ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "PICKUP", "TOGGLE", "DONE"]:
        assert action in prompt, f"'{action}' missing from prompt"


def test_build_prompt_ppo_suggestion():
    env = _make_fake_env()
    prompt = env.build_prompt(ppo_action=2)   # 2 = FORWARD
    assert "Autopilot suggests: FORWARD" in prompt


def test_build_prompt_no_suggestion_when_none():
    assert "Autopilot" not in _make_fake_env().build_prompt(ppo_action=None)


def test_build_prompt_ends_with_action_colon():
    assert _make_fake_env().build_prompt().strip().endswith("Action:")


def test_build_prompt_facing_direction():
    for i, name in enumerate(["EAST", "SOUTH", "WEST", "NORTH"]):
        env = _make_fake_env(agent_dir=i)
        assert f"Facing: {name}" in env.build_prompt()


def test_build_prompt_has_key_no():
    assert "Has key: NO" in _make_fake_env(has_key=False).build_prompt()


def test_build_prompt_has_key_yes():
    assert "Has key: YES" in _make_fake_env(has_key=True).build_prompt()


def test_build_prompt_step_counter():
    env = _make_fake_env(step=42, size=5)
    assert "42/250" in env.build_prompt()


def test_build_prompt_door_locked():
    env = _make_fake_env(door_open=False)
    assert "Door: LOCKED" in env.build_prompt()


def test_build_prompt_door_open():
    env = _make_fake_env(door_open=True)
    assert "Door: OPEN" in env.build_prompt()


# ---------------------------------------------------------------------------
# Unit tests: parse_action
# ---------------------------------------------------------------------------

def test_parse_all_exact_actions():
    assert parse_action("TURN_LEFT")  == 0
    assert parse_action("TURN_RIGHT") == 1
    assert parse_action("FORWARD")    == 2
    assert parse_action("PICKUP")     == 3
    assert parse_action("DROP")       == 4
    assert parse_action("TOGGLE")     == 5
    assert parse_action("DONE")       == 6


def test_parse_case_insensitive():
    assert parse_action("turn_left")  == 0
    assert parse_action("  forward ") == 2
    assert parse_action("toggle")     == 5


def test_parse_abbreviations():
    assert parse_action("LEFT")  == 0
    assert parse_action("RIGHT") == 1
    assert parse_action("FWD")   == 2
    assert parse_action("PICK")  == 3


def test_parse_longest_first_no_prefix_collision():
    """TURN_RIGHT must be found before RIGHT; PICKUP must be found before PICK."""
    assert parse_action("TURN_RIGHT") == 1
    assert parse_action("TURN_LEFT")  == 0
    assert parse_action("PICKUP")     == 3


def test_parse_action_embedded_in_text():
    assert parse_action("I choose FORWARD") == 2
    assert parse_action("Action: TOGGLE")   == 5


def test_parse_invalid_returns_none():
    assert parse_action("nonsense") is None
    assert parse_action("")         is None
    assert parse_action("   ")      is None


# ---------------------------------------------------------------------------
# Integration tests — require gymnasium + minigrid
# ---------------------------------------------------------------------------

pytest.importorskip("gymnasium", reason="gymnasium not installed")
pytest.importorskip("minigrid", reason="minigrid not installed")


@pytest.fixture(scope="module")
def real_env():
    env = DoorKeyEnv(size=5)
    yield env
    env.close()


def test_real_obs_shape(real_env):
    obs, _ = real_env.reset(seed=0)
    assert obs.shape == (147,)
    assert obs.dtype == np.float32


def test_real_action_space(real_env):
    assert real_env.action_space.n == 7


def test_real_step_returns_correct_types(real_env):
    obs, _ = real_env.reset(seed=1)
    obs2, reward, terminated, truncated, info = real_env.step(2)  # FORWARD
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_real_render_ascii(real_env):
    real_env.reset(seed=2)
    lines = real_env.render_view_ascii().split("\n")
    assert len(lines) == 7
    assert all(len(l) == 7 for l in lines)


def test_real_build_prompt_structure(real_env):
    real_env.reset(seed=3)
    prompt = real_env.build_prompt(ppo_action=2)
    assert "TURN_LEFT" in prompt
    assert "TOGGLE" in prompt
    assert "Autopilot suggests: FORWARD" in prompt
    assert prompt.strip().endswith("Action:")


def test_real_has_key_initially_false(real_env):
    real_env.reset(seed=4)
    assert real_env.has_key is False


def test_real_mc_uncertainty(real_env):
    from stable_baselines3 import PPO

    from ask.uncertainty.entropy import compute_mc_uncertainties
    from ask.utils.ppo import DropoutActorCriticPolicy

    tiny_env = DoorKeyEnv(size=5)
    model = PPO(
        DropoutActorCriticPolicy,
        tiny_env,
        policy_kwargs={"net_arch": [64, 64], "dropout_rate": 0.2},
        verbose=0,
        device="cpu",
    )
    tiny_env.close()

    obs, _ = real_env.reset(seed=5)
    total, _, _, probs = compute_mc_uncertainties(model, obs, n_samples=5)
    assert total >= 0.0
    assert probs.shape == (7,)
