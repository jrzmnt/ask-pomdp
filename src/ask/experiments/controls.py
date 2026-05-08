from __future__ import annotations


class EntropyGate:
    """
    Uncertainty gate: queries the SLM when policy entropy exceeds a threshold.
    """

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def should_query(self, entropy: float) -> bool:
        return entropy > self.threshold
