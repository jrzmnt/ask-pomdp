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
    from eval_ppo_slm import build_prompt
    env = FourRoomsEnv()
    env.reset(seed=0)
    prompt = build_prompt(env, ppo_action=2)
    assert "FORWARD" in prompt
    assert "TURN_LEFT" in prompt
    assert "TURN_RIGHT" in prompt
    env.close()


def test_parse_action():
    from eval_ppo_slm import parse_action
    assert parse_action("FORWARD") == 2
    assert parse_action("TURN_LEFT") == 0
    assert parse_action("TURN_RIGHT") == 1
    assert parse_action("  forward  ") == 2
    assert parse_action("nonsense") is None
    assert parse_action("") is None
