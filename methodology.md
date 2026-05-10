# Methodology — ASK-POMDP

## Overview

We evaluate the ASK framework — uncertainty-gated language assistance for reinforcement learning — under partial observability. Three agent types are compared across two environments: a PPO baseline, an SLM-only baseline, and the ASK method (PPO + SLM gated by MC Dropout uncertainty). All agents are evaluated on the same fixed test seeds to ensure fair comparison.

---

## Environments

### FourRooms (MiniGrid)

- **Task:** navigate a 19×19 grid with four rooms connected by doorways to reach a goal.
- **Partial observability:** agent receives only a 7×7 egocentric view (147-dimensional flat vector after normalization). Walls, doors, and unseen cells are included; the full map is never observed.
- **Action space:** TURN_LEFT (0), TURN_RIGHT (1), FORWARD (2).
- **Reward:** +1 on goal, 0 otherwise (sparse). Episode terminates at goal or `max_steps=500`.
- **Metric:** mean reward and episode length.

### HigherLower (POPGym)

- **Task:** predict whether the next card drawn is higher or lower than the current one, across a full 52-card deck.
- **Partial observability:** at each step the agent sees only the current card rank (discrete int 0–12). The full deck history is not provided in the observation — the agent must reason about it.
- **Action space:** HIGHER (0), LOWER (1).
- **Reward:** +1/52 for correct prediction, −1/52 for incorrect, 0 for a push (equal rank). An episode spans one full deck shuffle.
- **Metric:** mean reward and accuracy (fraction of correct decisions).
- **Memory requirement:** optimal play requires card counting — tracking all previously seen cards. The SLM prompt exposes this information explicitly; PPO must learn it implicitly via memory in the policy.

---

## Episode Splits

All splits are non-overlapping and defined solely by the seed passed to `env.reset(seed=s)`.

| Split      | Seeds   | Size | Purpose                              |
|------------|---------|------|--------------------------------------|
| Validation | 0–99    | 100  | Optuna threshold search (τ) for ASK  |
| Test       | 100–199 | 100  | Final evaluation of **all** agents   |

PPO, SLM-only, and ASK are all evaluated on the same test seeds (100–199). The validation set (0–99) is used only for Optuna and is never reported as a final result.

---

## Agents

### PPO (baseline)

- Policy: `DropoutActorCriticPolicy` — standard MLP actor-critic with MC Dropout support.
- Architecture: `net_arch = [256, 256]` (shared MLP), `dropout_rate = 0.2`.
- Trained with PPO (SB3) for **2M steps** on FourRooms and **500K steps** on HigherLower.
- Optimizer: Adam, `lr = 3e-4`; `n_steps = 2048`, `batch_size = 64`.
- At test time: deterministic action, dropout **disabled** (`mlp_extractor.eval()`).

#### PPO Optimality Ablation

To study how ASK behaves as PPO quality degrades, we snapshot models at reward thresholds during training using `RewardThresholdCheckpointCallback`. Each snapshot is saved the first time mean eval reward crosses a threshold, guaranteeing qualitative diversity rather than arbitrary step intervals.

| Environment | Thresholds |
|-------------|------------|
| FourRooms   | 0.1, 0.3, 0.5, 0.7 |
| HigherLower | 0.1, 0.2, 0.3, 0.4 |

Each checkpoint is evaluated with all three agent types. The x-axis in the ablation charts is the checkpoint reward level.

Checkpoint metadata saved alongside each `.zip`: `reward_threshold`, `eval_reward`, `training_steps`, `n_eval_episodes`, `saved_at`.

---

### SLM-only (baseline)

- Models: **Qwen/Qwen3.5-2B** and **Qwen/Qwen3.5-4B** (off-the-shelf, no fine-tuning).
- Thinking mode disabled: `enable_thinking=False` in `apply_chat_template`.
- At each step: full prompt (see below) → parse one-word action.
- No PPO suggestion in the prompt (pure LM decision).
- Invalid output → fallback to FORWARD (FourRooms) or HIGHER (HigherLower).
- **Intervention Rate = 1.0 by construction** (SLM called at every step).

---

### ASK (main method)

PPO and SLM are combined via uncertainty-gated switching:

1. Compute the PPO action deterministically (dropout off).
2. Run **N = 30 MC Dropout forward passes** (dropout enabled in `mlp_extractor`) to estimate action distribution uncertainty.
3. If `total_entropy ≥ τ` → query the SLM, including PPO's suggestion in the prompt.
4. If SLM returns a valid action, use it (overwrite); otherwise keep PPO's action.

The **Intervention Rate** (IR = `slm_called / steps`) measures how often ASK defers to the SLM. The **Overwrite Rate** (OR = `slm_overwrites / steps`) measures how often SLM actually changed the PPO action.

---

## Uncertainty Estimation (MC Dropout)

```
total_entropy   = H(E_θ[p(a|o)])    — entropy of mean distribution (bits, log₂)
aleatoric       = E_θ[H(p(a|o))]    — mean entropy per MC sample
epistemic       = total − aleatoric  — BALD approximation
```

The `mlp_extractor` is set to `train()` during the N passes and restored to `eval()` after. The policy head itself remains in eval mode. `total_entropy` is the gating signal.

---

## Threshold Selection (Optuna)

- Sampler: TPE, 15 trials, search space `τ ∈ [0.1, 2.0]` (uniform).
- Objective: maximize mean reward (FourRooms) or mean accuracy (HigherLower) on validation seeds 0–99.
- Storage: `sqlite:///optuna.db` (resumable with `load_if_exists=True`).
- Best τ saved to `results/thresholds.json` (FourRooms) and `higher_lower/results/thresholds.json` (HigherLower).
- **SLM and PPO model are loaded once** before the Optuna study; the same objects are reused across all 15 trials and the final evaluation pass (avoids repeated `torch.compile` overhead).
- Structured key format: `{env}_{model}` for main runs, `{env}_{model}_{ckpt_tag}` for checkpoint ablation.

---

## Prompt Design

### FourRooms

```
Grid maze (A=you  .=floor  #=wall  G=goal  ?=unseen  D=door).
Facing NORTH. Output one word: TURN_LEFT, TURN_RIGHT, or FORWARD.

#######
#.....#
#..A..#     ← 7×7 egocentric ASCII view
...
Goal: visible (3 ahead, 2 right). Path ahead: passable.
[PPO suggests: FORWARD]   ← ASK only; omitted for SLM-only
```

### HigherLower

```
Higher/Lower card game (A=lowest, K=highest).
Output one word: HIGHER or LOWER.

Card=A, 48 higher, 0 lower → HIGHER      ← 6 few-shot examples
Card=4, 39 higher, 8 lower → HIGHER
...
Card=9, 8 higher, 38 lower → LOWER

Card=7, 22 higher, 26 lower →[PPO suggests: LOWER]   ← ASK only
```

The HigherLower prompt includes the remaining deck composition (cards above and below current rank), enabling explicit card-counting reasoning. Few-shot examples are fixed and not derived from the episode.

---

## Metrics

| Metric | Definition | PPO | SLM | ASK |
|--------|-----------|:---:|:---:|:---:|
| Reward | episode return | ✓ | ✓ | ✓ |
| Accuracy | fraction correct decisions (HL only) | ✓ | ✓ | ✓ |
| Episode Length | steps until termination (FR only) | ✓ | ✓ | ✓ |
| IR | `slm_called / steps` | 0 | 1.0 | computed |
| OR | `slm_overwrites / steps` | 0 | — | computed |
| SLM Valid Rate | `valid_outputs / slm_called` | — | ✓ | ✓ |

All metrics reported as mean ± std over 100 test episodes (seeds 100–199).

---

## Reproducibility

| Item | Value |
|------|-------|
| Global seed | 42 (Python, NumPy, PyTorch, SB3) |
| Test seeds | 100–199 (fixed, logged in W&B config) |
| Val seeds | 0–99 (Optuna only) |
| PPO checkpoints | `runs/ppo/checkpoints/model_reward_*.zip` + `*_meta.json` |
| HL checkpoints | `runs/higher_lower/checkpoints/model_reward_*.zip` + `*_meta.json` |
| Optuna DB | `optuna.db` (resumable) |
| Thresholds | `results/thresholds.json`, `higher_lower/results/thresholds.json` |
| Result CSVs | `results/*.csv`, `higher_lower/results/*.csv` |
| LM weights | Hugging Face Hub — no fine-tuning, no local weights stored |

---

## W&B Organization

| Group | Runs |
|-------|------|
| `fourrooms` | PPO baseline, SLM-only (2B, 4B), ASK (2B, 4B) — FourRooms main results |
| `fourrooms_checkpoints` | Checkpoint ablation — FourRooms (4 PPO optimality levels × 3 agents) |
| `higherlower` | PPO baseline, SLM-only (2B, 4B), ASK (2B, 4B) — HigherLower main results |
| `higherlower_checkpoints` | Checkpoint ablation — HigherLower (4 PPO optimality levels × 3 agents) |

All runs log: summary metrics to `run.summary` and full config (model path, τ, N MC samples, seed splits, n_episodes). W&B project: `ask-pomdp-v2`.
