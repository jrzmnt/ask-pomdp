"""
HigherLower environment wrapper with card-counting state for prompt building.

Observation (from POPGym): discrete int 0–12 (current card rank index).
Actions: 0 = HIGHER, 1 = LOWER.
Reward: +1/52 correct, -1/52 incorrect, 0 push.

The wrapper tracks the full history of seen cards so the SLM prompt can
include remaining deck composition — enabling explicit card-counting reasoning.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from popgym.envs.higher_lower import HigherLower

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}  # A=0 ... K=12

ACTIONS_STR = ["HIGHER", "LOWER"]
_STR_TO_ACTION = {"HIGHER": 0, "LOWER": 1, "HIGH": 0, "LOW": 1}


class HigherLowerEnv(gym.Wrapper):
    """Gymnasium Wrapper around POPGym HigherLower with card-history tracking."""

    def __init__(self, num_decks: int = 1):
        super().__init__(HigherLower(num_decks=num_decks))
        self.num_decks = num_decks
        self._seen: List[str] = []
        self._current_card: str = ""

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[int, Dict]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._seen = []
        self._current_card = RANKS[obs]
        self._seen.append(self._current_card)
        return obs, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_card = RANKS[obs]
        self._seen.append(self._current_card)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def remaining_counts(self) -> Counter:
        """Counts of each rank still in the deck (not yet revealed)."""
        seen = Counter(self._seen)
        total = Counter({r: 4 * self.num_decks for r in RANKS})
        remaining = total - seen
        return remaining

    def cards_above(self, card: str) -> int:
        """Number of remaining cards strictly above `card`."""
        idx = RANK_VALUES[card]
        rem = self.remaining_counts()
        # exclude the current card itself from remaining
        rem[card] = max(0, rem[card] - 1)
        return sum(rem[r] for r in RANKS if RANK_VALUES[r] > idx)

    def cards_below(self, card: str) -> int:
        """Number of remaining cards strictly below `card`."""
        idx = RANK_VALUES[card]
        rem = self.remaining_counts()
        rem[card] = max(0, rem[card] - 1)
        return sum(rem[r] for r in RANKS if RANK_VALUES[r] < idx)

    def cards_equal(self, card: str) -> int:
        """Number of remaining cards with the same rank as `card`."""
        rem = self.remaining_counts()
        return max(0, rem[card] - 1)

    def build_prompt(self, ppo_action: Optional[int] = None) -> str:
        card = self._current_card
        above = self.cards_above(card)
        below = self.cards_below(card)
        equal = self.cards_equal(card)
        ppo_line = f"\nAutopilot suggests: {ACTIONS_STR[ppo_action]}" if ppo_action is not None else ""

        return f"""\
You are a card game decision policy.
Your task is to choose exactly ONE action.

VALID ACTIONS:
HIGHER
LOWER

RULES:
- Do NOT explain.
- Do NOT add text or markdown.
- Choose HIGHER if more cards remaining are above the current card.
- Choose LOWER if more cards remaining are below the current card.
- If the autopilot suggestion is consistent with the card counts, follow it.

STATE:
Current card: {card}
Cards remaining higher: {above}
Cards remaining lower: {below}
Cards remaining equal: {equal}{ppo_line}

Examples:
Card=A, 48 higher, 0 lower → HIGHER
Card=4, 39 higher, 8 lower → HIGHER
Card=9, 8 higher, 38 lower → LOWER
Card=K, 0 higher, 48 lower → LOWER

OUTPUT FORMAT (MANDATORY):
HIGHER or LOWER

Action: """


