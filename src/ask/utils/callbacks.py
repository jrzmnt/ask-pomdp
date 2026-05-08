from stable_baselines3.common.callbacks import EvalCallback


class EvalCallbackWithEvalMode(EvalCallback):
    """EvalCallback que desativa o dropout durante a avaliação."""

    def _on_step(self) -> bool:
        self.model.policy.set_training_mode(False)
        result = super()._on_step()
        self.model.policy.set_training_mode(True)
        return result
