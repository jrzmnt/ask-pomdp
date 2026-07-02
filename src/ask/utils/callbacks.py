from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class EvalCallbackWithEvalMode(EvalCallback):
    """EvalCallback que desativa o dropout durante a avaliação."""

    def _on_step(self) -> bool:
        self.model.policy.set_training_mode(False)
        result = super()._on_step()
        self.model.policy.set_training_mode(True)
        return result


class RewardThresholdCheckpointCallback(EvalCallbackWithEvalMode):
    """Saves a checkpoint the first time eval reward crosses each threshold.

    Inherits periodic eval + best-model saving from EvalCallback.
    On each crossing: saves model_reward_NNN.zip + model_reward_NNN_meta.json
    with eval_reward, training_steps, threshold, and timestamp.

    Thresholds example:
      FourRooms   → [0.1, 0.3, 0.5, 0.7]
      HigherLower → [0.1, 0.2, 0.3, 0.4]
    """

    def __init__(self, thresholds: list[float], checkpoint_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending = sorted(thresholds)
        self._ckpt_dir = Path(checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward > -np.inf and self._pending:
            crossed = [t for t in self._pending if self.last_mean_reward >= t]
            for t in crossed:
                name = f"model_reward_{int(t * 100):03d}"
                self.model.save(str(self._ckpt_dir / name))

                meta = {
                    "checkpoint_name":  name,
                    "reward_threshold": t,
                    "eval_reward":      float(self.last_mean_reward),
                    "training_steps":   self.num_timesteps,
                    "n_eval_episodes":  self.n_eval_episodes,
                    "saved_at":         datetime.now().isoformat(),
                }
                with open(self._ckpt_dir / f"{name}_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

                print(
                    f"  [ckpt] reward={self.last_mean_reward:.3f} >= {t:.2f} "
                    f"(step {self.num_timesteps:,}) → {name}.zip"
                )
                self._pending.remove(t)
        return result
