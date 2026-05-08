from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


def policy_entropy(model, obs: np.ndarray) -> float:
    """Entropy of the PPO action distribution for a single observation."""
    if not isinstance(model.policy, ActorCriticPolicy):
        raise TypeError("Expects an ActorCriticPolicy.")

    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        entropy = distribution.entropy()

    return float(entropy.mean().cpu().item())


def compute_mc_uncertainties(model, obs: np.ndarray, n_samples: int = 30):
    """
    MC Dropout uncertainty decomposition.

    Returns (total, aleatoric, epistemic, mean_probs).
    Keeps mlp_extractor in train mode for stochastic forward passes.
    """
    model.policy.mlp_extractor.train()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)

    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            dist = model.policy.get_distribution(obs_tensor)
            probs_list.append(dist.distribution.probs.squeeze(0).cpu())

    probs = torch.stack(probs_list)         # (n_samples, n_actions)
    mean_probs = probs.mean(dim=0)

    log = torch.log2
    total = -(mean_probs * log(mean_probs + 1e-10)).sum()
    aleatoric = -(probs * log(probs + 1e-10)).sum(dim=1).mean()
    epistemic = (probs * log((probs + 1e-10) / (mean_probs + 1e-10))).sum(dim=1).mean()

    model.policy.mlp_extractor.eval()

    return float(total), float(aleatoric), float(epistemic), mean_probs
