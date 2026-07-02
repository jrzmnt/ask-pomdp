"""
Diagnostic: test raw SLM outputs on HigherLower prompts.

Compares prompt strategies on 6 hand-crafted unambiguous cases.

Usage:
  python higher_lower/scripts/test_slm_outputs.py --slm 1.5b
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

import torch
from ask.slm.model import load_slm
from higher_lower.env import ACTIONS_STR, _STR_TO_ACTION

QWEN_MODELS = {
    "0.5b":       "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b":       "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

CASES = [
    ("A",  48,  0, 3, "HIGHER"),
    ("2",  47,  0, 3, "HIGHER"),
    ("5",  36, 12, 3, "HIGHER"),
    ("7",  24, 24, 3, "HIGHER"),  # tie
    ("Q",   4, 44, 3, "LOWER"),
    ("K",   0, 48, 3, "LOWER"),
]


# ── prompt variants ────────────────────────────────────────────────────────────

def prompt_baseline(card, above, below, equal):
    total = above + below + equal
    return (
        f"You are playing a Higher/Lower card game (A=lowest, K=highest).\n\n"
        f"Current card: {card}\n"
        f"Remaining ({total}): {above} higher, {below} lower, {equal} same.\n\n"
        f"Output EXACTLY one word: HIGHER or LOWER"
    )


def prompt_few_shot(card, above, below, equal):
    total = above + below + equal
    return (
        f"Higher/Lower card game (A=lowest, K=highest).\n"
        f"Output one word: HIGHER or LOWER.\n\n"
        f"Card=A, 48 higher, 0 lower → HIGHER\n"
        f"Card=4, 39 higher, 8 lower → HIGHER\n"
        f"Card=6, 28 higher, 18 lower → HIGHER\n"
        f"Card=9, 8 higher, 38 lower → LOWER\n"
        f"Card=Q, 4 higher, 44 lower → LOWER\n"
        f"Card=K, 0 higher, 48 lower → LOWER\n\n"
        f"Card={card}, {above} higher, {below} lower → "
    )


def prompt_cot_last(card, above, below, equal):
    """CoT but parsed from the LAST match in the output."""
    total = above + below + equal
    return (
        f"Higher/Lower card game (A=lowest, K=highest).\n"
        f"Card: {card}. Remaining {total} cards: {above} higher, {below} lower.\n\n"
        f"Reasoning: compare the counts. "
    )


def prompt_answer_last(card, above, below, equal):
    """Force the word to appear at the end by completing a sentence."""
    total = above + below + equal
    larger = above if above >= below else below
    direction = "higher" if above >= below else "lower"
    return (
        f"Higher/Lower card game (A=lowest, K=highest).\n"
        f"Card: {card}. Of {total} remaining cards, {larger} are {direction}.\n"
        f"The action that wins most often is: "
    )


def prompt_env(card, above, below, equal):
    """Exact prompt from env.build_prompt() — what actually runs in eval."""
    from higher_lower.env import HigherLowerEnv
    env = HigherLowerEnv.__new__(HigherLowerEnv)
    env._current_card = card
    env.num_decks = 1
    env._seen = [card]  # minimal state
    # Override counts directly
    env.cards_above = lambda c: above
    env.cards_below = lambda c: below
    env.cards_equal = lambda c: equal
    return env.build_prompt()


VARIANTS = {
    "env_prompt":  (prompt_env,         {"max_tokens": 10}, "first"),
    "baseline":    (prompt_baseline,    {"max_tokens": 5},  "first"),
    "few_shot":    (prompt_few_shot,    {"max_tokens": 5},  "first"),
    "cot_last":    (prompt_cot_last,    {"max_tokens": 40}, "last"),
    "answer_last": (prompt_answer_last, {"max_tokens": 10}, "last"),
}


# ── parsing ────────────────────────────────────────────────────────────────────

def parse_first(text: str):
    t = text.strip().upper()
    for key, val in _STR_TO_ACTION.items():
        if key in t:
            return ACTIONS_STR[val]
    return None


def parse_last(text: str):
    """Return the last occurrence of HIGHER or LOWER in the text."""
    matches = list(re.finditer(r'\b(HIGHER|LOWER|HIGH|LOW)\b', text.upper()))
    if not matches:
        return None
    last = matches[-1].group()
    return ACTIONS_STR[_STR_TO_ACTION[last]]


def _slm_cfg(model_name: str) -> dict:
    return {
        "provider": "hf",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "float16",
    }


# ── test ───────────────────────────────────────────────────────────────────────

def test_model(slm_key: str) -> None:
    model_name = QWEN_MODELS[slm_key]
    print(f"\n{'='*70}")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    slm = load_slm(_slm_cfg(model_name))
    scores = {}

    for variant_name, (prompt_fn, decoding, parse_mode) in VARIANTS.items():
        parse_fn = parse_last if parse_mode == "last" else parse_first
        print(f"\n── {variant_name} (max_tokens={decoding['max_tokens']}, parse={parse_mode}) ──")
        print(f"{'Card':<5} {'↑':>4} {'↓':>4}  {'Correct':>7}  {'Raw output':<40}  {'Parsed':>7}  OK")
        print("-" * 78)
        correct = 0
        for card, above, below, equal, expected in CASES:
            prompt = prompt_fn(card, above, below, equal)
            raw = slm.generate(prompt, decoding).text.strip()
            parsed = parse_fn(raw)
            ok = "✓" if parsed == expected else "✗"
            if parsed == expected:
                correct += 1
            raw_repr = repr(raw[:45]) if len(raw) > 45 else repr(raw)
            print(f"{card:<5} {above:>4} {below:>4}  {expected:>7}  {raw_repr:<40}  {str(parsed):>7}  {ok}")
        scores[variant_name] = correct
        print(f"  → {correct}/{len(CASES)}")

    print(f"\n{'─'*70}")
    print(f"  Score summary for {slm_key}:")
    for v, s in scores.items():
        bar = "█" * s + "░" * (len(CASES) - s)
        print(f"    {v:<14} {bar}  {s}/{len(CASES)}")
    best = max(scores, key=scores.get)
    print(f"  Best variant: {best} ({scores[best]}/{len(CASES)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slm", choices=list(QWEN_MODELS.keys()), default="1.5b")
    args = p.parse_args()
    test_model(args.slm)


if __name__ == "__main__":
    main()
