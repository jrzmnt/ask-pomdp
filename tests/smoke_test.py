"""
Smoke tests: verify the full pipeline runs end-to-end without errors.
No GPU required. SLM calls are mocked.
Runtime: ~60-90 seconds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from ask.envs.fourrooms import FourRoomsEnv
from ask.experiments.controls import EntropyGate
from ask.slm.model import SLMOutput
from ask.uncertainty.entropy import compute_mc_uncertainties
from ask.utils.ppo import DropoutActorCriticPolicy


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Train a minimal PPO model (128 steps) for reuse across tests."""
    tmp = tmp_path_factory.mktemp("model")
    env = FourRoomsEnv()
    model = PPO(
        policy=DropoutActorCriticPolicy,
        env=env,
        policy_kwargs={"dropout_rate": 0.2, "net_arch": {"pi": [64], "vf": [64]}},
        n_steps=64,
        batch_size=32,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=128)
    path = str(tmp / "model")
    model.save(path)
    env.close()
    return path, model


@pytest.fixture
def mock_slm():
    slm = MagicMock()
    slm.generate.return_value = SLMOutput(
        text="FORWARD", logits=torch.zeros(10), cost=1.0
    )
    return slm


# =============================================================================
# Environment
# =============================================================================

def test_env_obs_shape():
    env = FourRoomsEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (147,), f"Expected (147,), got {obs.shape}"
    env.close()


def test_env_action_space():
    env = FourRoomsEnv()
    env.reset(seed=0)
    assert env.action_space.n == 3
    env.close()


def test_env_step():
    env = FourRoomsEnv()
    env.reset(seed=0)
    for action in [0, 1, 2]:
        env.reset(seed=action)
        obs, r, term, trunc, _ = env.step(action)
        assert obs.shape == (147,)
        assert isinstance(r, (int, float))
    env.close()


def test_env_render_ascii():
    env = FourRoomsEnv()
    env.reset(seed=0)
    view = env.render_view_ascii()
    lines = view.split("\n")
    assert len(lines) == 7
    assert all(len(l) == 7 for l in lines)
    env.close()


def test_env_agent_dir():
    env = FourRoomsEnv()
    env.reset(seed=0)
    assert env.agent_dir in {0, 1, 2, 3}
    env.close()


def test_env_agent_pos_abs():
    env = FourRoomsEnv()
    env.reset(seed=0)
    x, y = env.agent_pos_abs
    w, h = env._env.unwrapped.width, env._env.unwrapped.height
    assert 0 <= x < w and 0 <= y < h
    env.close()


# =============================================================================
# PPO + uncertainty
# =============================================================================

def test_ppo_predict(tiny_model):
    _, model = tiny_model
    env = FourRoomsEnv()
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in {0, 1, 2}
    env.close()


def test_mc_uncertainty_shape(tiny_model):
    _, model = tiny_model
    env = FourRoomsEnv()
    obs, _ = env.reset(seed=0)
    total, alea, epis, probs = compute_mc_uncertainties(model, obs, n_samples=5)
    assert total >= 0.0
    assert alea >= 0.0
    assert epis >= 0.0
    assert probs.shape == (3,)
    env.close()


def test_entropy_gate():
    gate = EntropyGate(threshold=0.5)
    assert gate.should_query(1.0) is True
    assert gate.should_query(0.5) is False  # strictly greater
    assert gate.should_query(0.1) is False


# =============================================================================
# Eval functions (W&B mocked out)
# =============================================================================

@patch("wandb.init")
@patch("wandb.run", new_callable=MagicMock)
def test_eval_ppo(mock_run, mock_init, tiny_model):
    model_path, _ = tiny_model
    from eval_ppo_slm import eval_ppo
    summary, logs = eval_ppo(model_path, n_episodes=3)
    assert "mean_reward" in summary
    assert "mean_length" in summary
    assert summary["n_episodes"] == 3
    assert len(logs) == 3


def test_eval_ask_loop(tiny_model, mock_slm):
    _, model = tiny_model
    from eval_ppo_slm import eval_ask
    mean_reward, logs = eval_ask(
        model=model,
        slm=mock_slm,
        threshold=0.5,
        n_episodes=3,
        seed_offset=0,
        n_mc_samples=5,
    )
    assert isinstance(mean_reward, float)
    assert len(logs) == 3
    for log in logs:
        assert "IR" in log
        assert "OR" in log
        assert "reward" in log
        assert "length" in log
        assert log["length"] > 0
        assert 0.0 <= log["IR"] <= 1.0
        assert 0.0 <= log["OR"] <= 1.0


def test_eval_ask_always_ask(tiny_model, mock_slm):
    """τ=0 → SLM is queried every step (IR ≈ 1.0)."""
    _, model = tiny_model
    from eval_ppo_slm import eval_ask
    _, logs = eval_ask(
        model=model, slm=mock_slm, threshold=0.0,
        n_episodes=2, seed_offset=10, n_mc_samples=3,
    )
    for log in logs:
        assert log["IR"] == pytest.approx(1.0)


def test_eval_ask_never_ask(tiny_model, mock_slm):
    """τ=∞ → SLM is never queried (IR = 0.0)."""
    _, model = tiny_model
    from eval_ppo_slm import eval_ask
    _, logs = eval_ask(
        model=model, slm=mock_slm, threshold=999.0,
        n_episodes=2, seed_offset=20, n_mc_samples=3,
    )
    for log in logs:
        assert log["IR"] == pytest.approx(0.0)
        assert log["OR"] == pytest.approx(0.0)
    # SLM should never have been called
    mock_slm.generate.assert_not_called()


def test_prompt_build():
    from eval_ppo_slm import build_prompt, new_episode_state, update_episode_state
    env = FourRoomsEnv()
    env.reset(seed=0)
    prompt = build_prompt(env, None, ppo_action=2, prompt_style="basic")
    assert "FORWARD" in prompt
    assert "TURN_LEFT" in prompt
    assert "TURN_RIGHT" in prompt
    p2 = build_prompt(env, None, None, prompt_style="enriched")
    assert "ACTION PREVIEW" in p2
    st = new_episode_state()
    update_episode_state(st, env, None)
    p3 = build_prompt(env, st, None, prompt_style="stateful_min")
    assert "World position" in p3
    env.close()


def test_parse_action():
    from eval_ppo_slm import parse_action
    assert parse_action("FORWARD") == 2
    assert parse_action("TURN_LEFT") == 0
    assert parse_action("TURN_RIGHT") == 1
    assert parse_action("  forward  ") == 2
    assert parse_action("nonsense") is None
    assert parse_action("") is None


def test_parse_action_rationale_prefers_action_line():
    from eval_ppo_slm import parse_action
    text = "Reason: I want to go left first\nAction: TURN_RIGHT"
    assert parse_action(text, rationale=True) == 1
    text2 = "Reason: mention TURN_LEFT in text\nAction: FORWARD"
    assert parse_action(text2, rationale=True) == 2


# =============================================================================
# HigherLower prompt enrichment
# =============================================================================


def test_hl_prompt_styles():
    from higher_lower.env import HigherLowerEnv

    env = HigherLowerEnv()
    env.reset(seed=0)
    basic = env.build_prompt(prompt_style="basic")
    assert "HIGHER" in basic and "LOWER" in basic
    assert "Cards remaining strictly higher" in basic
    assert "Recommended" not in basic

    enriched = env.build_prompt(prompt_style="enriched")
    assert "Recommended" in enriched
    assert "P(next strictly higher" in enriched

    env.reset(seed=1)
    for _ in range(3):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    stateful = env.build_prompt(prompt_style="stateful", prompt_history=4)
    assert "Recent decisions" in stateful
    assert "Win streak" in stateful
    env.close()


def test_hl_parse_action_rationale():
    from higher_lower.eval import parse_action

    assert (
        parse_action("Reason: HIGHER looks tempting\nAction: LOWER", rationale=True) == 1
    )
    assert parse_action("LOWER", rationale=True) == 1
    assert parse_action("Reason: x\nAction: HIGHER", rationale=True) == 0


def test_hl_episode_state_history():
    from higher_lower.env import HigherLowerEnv

    env = HigherLowerEnv()
    env.reset(seed=0)
    assert len(env._actions) == 0
    assert len(env._outcomes) == 0
    env.step(0)
    assert len(env._actions) == 1
    assert len(env._outcomes) == 1
    assert len(env._history_cards) == 1
    assert len(env._seen) >= 2
    env.close()


# =============================================================================
# DoorKey prompt enrichment
# =============================================================================


def test_dk_env_agent_pos_abs_and_obs_cell():
    from door_key.env import DoorKeyEnv

    env = DoorKeyEnv(size=5)
    env.reset(seed=0)
    ap = env.agent_pos_abs
    w, h = env._env.unwrapped.width, env._env.unwrapped.height
    assert 0 <= ap[0] < w and 0 <= ap[1] < h
    hits = [(r, c) for r in range(7) for c in range(7) if env.obs_cell_to_world(r, c) == ap]
    assert len(hits) == 1
    env.close()


def test_dk_prompt_styles():
    from door_key.env import DoorKeyEnv
    from door_key.eval import build_prompt, new_episode_state, update_episode_state

    env = DoorKeyEnv(size=5)
    env.reset(seed=0)
    basic = build_prompt(env, None, prompt_style="basic")
    assert "TURN_LEFT" in basic and "PICKUP" in basic and "TOGGLE" in basic
    assert "CURRENT SUBTASK" in basic
    assert "ACTION PREVIEW" not in basic

    enriched = build_prompt(env, None, prompt_style="enriched")
    assert "ACTION PREVIEW" in enriched
    assert "Adjacent" in enriched
    assert "Longest clear ray" in enriched

    st = new_episode_state()
    update_episode_state(st, env, None)
    s_min = build_prompt(env, st, prompt_style="stateful_min")
    assert "World position" in s_min
    assert "Recent actions" in s_min

    s_full = build_prompt(env, st, prompt_style="stateful", prompt_map_radius=3)
    assert "DISCOVERED MAP" in s_full

    s_rat = build_prompt(env, st, prompt_style="stateful", rationale=True)
    assert "Reason:" in s_rat
    assert "Action:" in s_rat
    env.close()


def test_dk_parse_action_rationale():
    from door_key.eval import parse_action

    assert parse_action("PICKUP") == 3
    assert parse_action("TOGGLE") == 5
    assert parse_action("FORWARD") == 2
    text = "Reason: I should pickup the key soon\nAction: TURN_LEFT"
    assert parse_action(text, rationale=True) == 0
    text2 = "Reason: ahead is a wall\nAction: TURN_RIGHT"
    assert parse_action(text2, rationale=True) == 1


# =============================================================================
# RandomSLM (dice) baseline
# =============================================================================


def test_random_slm_returns_valid_tokens():
    from ask.slm.model import RandomSLM, SLMOutput

    tokens = ["A", "B", "C"]
    rnd = RandomSLM(tokens, seed=0)
    counts = {t: 0 for t in tokens}
    for _ in range(200):
        out = rnd.generate("ignored", {"max_tokens": 1})
        assert isinstance(out, SLMOutput)
        assert out.text in tokens
        counts[out.text] += 1
    # All tokens hit at least once with seed=0
    assert all(c > 0 for c in counts.values())


def test_random_slm_via_load_slm_dispatch():
    from ask.slm.model import load_slm, RandomSLM

    slm = load_slm({"provider": "random", "actions": ["FORWARD", "TURN_LEFT", "TURN_RIGHT"], "seed": 1})
    assert isinstance(slm, RandomSLM)
    text = slm.generate("p", None).text
    assert text in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}


def test_eval_ask_with_random_slm_fourrooms(tiny_model):
    """ASK loop runs end-to-end with the RandomSLM dice baseline."""
    from ask.slm.model import RandomSLM
    from eval_ppo_slm import ACTIONS_STR, eval_ask

    _, model = tiny_model
    slm = RandomSLM(ACTIONS_STR, seed=42)
    _, logs = eval_ask(
        model=model,
        slm=slm,
        threshold=0.0,  # ask every step → IR=1
        n_episodes=2,
        seed_offset=30,
        n_mc_samples=3,
    )
    assert len(logs) == 2
    for log in logs:
        assert log["IR"] == pytest.approx(1.0)
        assert 0.0 <= log["OR"] <= 1.0
        # RandomSLM always returns a valid token, so slm_valid_rate == 1
        assert log["slm_valid_rate"] == pytest.approx(1.0)


def test_dk_episode_state_updates():
    from door_key.env import DoorKeyEnv
    from door_key.eval import new_episode_state, update_episode_state

    env = DoorKeyEnv(size=5)
    env.reset(seed=0)
    st = new_episode_state()
    update_episode_state(st, env, None)
    assert st.last_pos == env.agent_pos_abs
    assert st.visits[env.agent_pos_abs] >= 1
    assert len(st.known_grid) > 0
    for a in (0, 1, 2):
        env.step(a)
        update_episode_state(st, env, a)
    assert len(st.actions) == 3
    env.close()
