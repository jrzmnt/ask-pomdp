"""
HigherLower environment wrapper with card-counting state for prompt building.

Observation (from POPGym): discrete int 0–12 (current card rank index).
Actions: 0 = HIGHER, 1 = LOWER.
Reward: +1/52 correct, -1/52 incorrect, 0 push.

The wrapper tracks the full history of seen cards so the SLM prompt can
include remaining deck composition — enabling explicit card-counting reasoning.
It also tracks per-episode actions and outcomes for memory-rich prompts.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from popgym.envs.higher_lower import HigherLower

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}  # A=0 ... K=12

ACTIONS_STR = ["HIGHER", "LOWER"]
_STR_TO_ACTION = {"HIGHER": 0, "LOWER": 1, "HIGH": 0, "LOW": 1}


def _outcome_from_reward(reward: float) -> int:
    """Map POPGym reward to {-1, 0, +1} (loss / push / win)."""
    return int(round(float(reward) * 52))


class HigherLowerEnv(gym.Wrapper):
    """Gymnasium Wrapper around POPGym HigherLower with card-history tracking."""

    def __init__(self, num_decks: int = 1):
        super().__init__(HigherLower(num_decks=num_decks))
        self.num_decks = num_decks
        self._seen: List[str] = []
        self._current_card: str = ""
        self._actions: List[int] = []
        self._outcomes: List[int] = []
        # For each step: card shown when deciding, action taken, outcome, next card revealed
        self._history_cards: List[str] = []
        self._history_next: List[str] = []

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[int, Dict]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._seen = []
        self._actions = []
        self._outcomes = []
        self._history_cards = []
        self._history_next = []
        self._current_card = RANKS[obs]
        self._seen.append(self._current_card)
        return obs, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        decision_card = self._current_card
        obs, reward, terminated, truncated, info = self.env.step(action)
        outcome = _outcome_from_reward(reward)
        self._actions.append(int(action))
        self._outcomes.append(outcome)
        self._current_card = RANKS[obs]
        self._seen.append(self._current_card)
        self._history_cards.append(decision_card)
        self._history_next.append(self._current_card)
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

    @property
    def episode_step(self) -> int:
        """Number of decisions taken this episode (0 after reset, before first step)."""
        return len(self._actions)

    def win_streak(self) -> int:
        n = 0
        for o in reversed(self._outcomes):
            if o > 0:
                n += 1
            else:
                break
        return n

    def loss_streak(self) -> int:
        n = 0
        for o in reversed(self._outcomes):
            if o < 0:
                n += 1
            else:
                break
        return n

    def _recent_history_block(self, prompt_history: int) -> str:
        if not self._actions:
            return "Recent decisions: (none yet — this is the first guess.)\n"
        lines: List[str] = []
        start = max(0, len(self._actions) - prompt_history)
        for i in range(start, len(self._actions)):
            c = self._history_cards[i]
            a = ACTIONS_STR[self._actions[i]]
            o = self._outcomes[i]
            oc = "WIN" if o > 0 else "LOSS" if o < 0 else "PUSH"
            nxt = self._history_next[i]
            lines.append(f"  On {c} chose {a} → {oc} (next card {nxt})")
        tail = "\n".join(lines)
        ws = self.win_streak()
        ls = self.loss_streak()
        streak = f"Win streak: {ws}  |  Loss streak: {ls}"
        return f"Recent decisions (last up to {prompt_history}):\n{tail}\n{streak}\n"

    def _enriched_block(self, card: str, above: int, below: int, equal: int) -> str:
        denom = above + below
        if denom <= 0:
            p_hi = p_lo = 0.5
            margin = 0
            recommended = "EITHER"
        else:
            p_hi = above / denom
            p_lo = below / denom
            margin = abs(above - below)
            if above > below:
                recommended = "HIGHER"
            elif below > above:
                recommended = "LOWER"
            else:
                recommended = "EITHER"
        total = above + below + max(equal, 0)
        p_push = (equal / total) if total > 0 else 0.0
        return f"""\
Count-derived hints (remaining unknown cards; current rank excluded from higher/lower pools):
P(next strictly higher | not push) ≈ {p_hi:.4f}
P(next strictly lower  | not push) ≈ {p_lo:.4f}
P(push / same rank) among remaining mass ≈ {p_push:.4f} (equal ranks left: {equal})
Recommended action from counts: {recommended}
Margin |higher − lower|: {margin}
"""

    def build_prompt(
        self,
        ppo_action: Optional[int] = None,
        *,
        prompt_style: str = "basic",
        rationale: bool = False,
        prompt_history: int = 8,
    ) -> str:
        allowed = ("basic", "enriched", "stateful")
        if prompt_style not in allowed:
            raise ValueError(f"prompt_style must be one of {allowed}, got {prompt_style!r}")

        card = self._current_card
        above = self.cards_above(card)
        below = self.cards_below(card)
        equal = self.cards_equal(card)
        ppo_line = f"\nAutopilot suggests: {ACTIONS_STR[ppo_action]}" if ppo_action is not None else ""

        rules_no_expl = (
            "- Do NOT explain.\n"
            "- Do NOT add text or markdown.\n"
        )
        rules_rationale = (
            "- First line: Reason: <one short line, max 12 words>.\n"
            "- Second line: Action: <HIGHER|LOWER>.\n"
            "- No other text or markdown.\n"
        )
        rules_core = (
            "- Choose HIGHER if more cards remaining are strictly above the current card.\n"
            "- Choose LOWER if more cards remaining are strictly below the current card.\n"
            "- If counts tie, either action is symmetric; use P(push) and recent luck as tie-breakers.\n"
        )
        rules_tail = (
            "- If the autopilot suggestion is consistent with the card counts, follow it.\n"
            if ppo_line
            else ""
        )

        state_block = f"""\
STATE:
Current card: {card}
Cards remaining strictly higher: {above}
Cards remaining strictly lower: {below}
Cards remaining equal rank: {equal}{ppo_line}
"""

        enriched = ""
        if prompt_style in ("enriched", "stateful"):
            enriched = self._enriched_block(card, above, below, equal)

        memory = ""
        if prompt_style == "stateful":
            memory = (
                f"Episode decision index: {self.episode_step} "
                f"(after this choice, {len(self._seen) - 1} cards have been seen including the opening card).\n"
                + self._recent_history_block(prompt_history)
            )

        if prompt_style == "basic":
            rules = rules_no_expl + rules_core + rules_tail
            out_fmt = "HIGHER or LOWER\n\nAction: "
        else:
            rules = (
                (rules_rationale if rationale else rules_no_expl)
                + rules_core
                + rules_tail
            )
            out_fmt = (
                "Reason: <short>\nAction: HIGHER or LOWER\n\n"
                if rationale
                else "HIGHER or LOWER\n\nAction: "
            )

        return f"""\
You are a card game decision policy (Higher / Lower vs the next card from a finite shoe).
Your task is to choose exactly ONE action.

VALID ACTIONS:
HIGHER
LOWER

RULES:
{rules}
{memory}{state_block}{enriched}
Examples:
Card=A, many more ranks above → HIGHER
Card=K, many more ranks below → LOWER
Counts tie with pushes possible → pick using push probability and recent streaks

OUTPUT FORMAT (MANDATORY):
{out_fmt}"""

