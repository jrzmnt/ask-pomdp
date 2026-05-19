# DoorKey — ASK Pipeline Guide

MiniGrid-DoorKey is a sequential POMDP: the agent must find a key, pick it up,
unlock a door, and reach the goal — all within a fixed step budget. The environment
is available in four difficulty levels (5×5 → 16×16).

---

## Seed Protocol (stable methodology)

| Split | Seeds | Purpose |
|-------|-------|---------|
| **Train** | 200–999 (800 maps) | `SeededDoorKeyEnv(seeds=TRAIN_SEEDS)` — never seen in val/test |
| **Val** | 0–99 (100 maps) | Optuna τ search — no overlap with test |
| **Test** | 100–199 (100 maps) | Final reported results |

Enforced automatically: `train.py` uses `SeededDoorKeyEnv` for both the training
env and the eval callback. `eval.py` drives seeds explicitly via `seed_offset`.

---

## SLM Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | `Qwen/Qwen3.5-2B` | Same as other envs |
| Thinking | **False** | Subtask is explicit in prompt; `max_tokens=10` |
| `max_tokens` | 10 | Response is a single action word (e.g. `FORWARD`) |
| Fallback | `FORWARD` (2) | Keeps agent moving on invalid SLM output |

Unlike Craftax (where thinking=True is needed for the crafting chain), DoorKey
already tells the model its current subtask verbatim — extended reasoning adds
latency without benefit.

---

## Scripts — Execution Order

### 1. Run smoke tests (no GPU needed)

```bash
uv run python -m pytest tests/smoke_test_doorkey.py -v --tb=short -k "not real_"
```

Expected: **29 passed** in < 30 s.

---

### 2. Train PPO — one size at a time

```bash
# 5×5  (~30 min CPU / ~10 min GPU)
uv run python door_key/train.py --size 5 --timesteps 500000

# 6×6  (~45 min CPU / ~15 min GPU)
uv run python door_key/train.py --size 6 --timesteps 500000

# 8×8  (~90 min CPU / ~30 min GPU)
uv run python door_key/train.py --size 8 --timesteps 1000000

# 16×16 (~4h CPU / ~1.5h GPU) — leave for last
uv run python door_key/train.py --size 16 --timesteps 2000000
```

Checkpoints saved to `runs/door_key/checkpoints_sX/` at rewards [0.3, 0.5, 0.7, 0.9].
Final model saved to `runs/door_key/model_sX.zip`.

Or run all sizes sequentially:

```bash
bash door_key/scripts/run_doorkey.sh --sizes 5 6 8 16 --timesteps 500000
```

---

### 3. Evaluate PPO baseline

```bash
# Per size
uv run python door_key/eval.py --mode ppo --size 5
uv run python door_key/eval.py --mode ppo --size 6
uv run python door_key/eval.py --mode ppo --size 8
uv run python door_key/eval.py --mode ppo --size 16
```

Output: `door_key/results/ppo_results_sX.json` + `ppo_episodes_sX.csv`

---

### 4. Evaluate SLM-only

```bash
uv run python door_key/eval.py --mode slm --slm qwen3.5-2b --size 5
uv run python door_key/eval.py --mode slm --slm qwen3.5-2b --size 6
uv run python door_key/eval.py --mode slm --slm qwen3.5-2b --size 8
uv run python door_key/eval.py --mode slm --slm qwen3.5-2b --size 16
```

Output: `door_key/results/slm_qwen35_2b_results_sX.json`

---

### 5. Evaluate ASK (Optuna τ search + test)

```bash
# Optuna finds best τ on val seeds (0–99), then evaluates on test seeds (100–199)
uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 5
uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 6
uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 8
uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 16
```

Best τ per size saved to `door_key/results/thresholds.json`.
Output: `door_key/results/ask_qwen35_2b_results_sX.json`

To fix τ manually (skip Optuna):

```bash
uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 5 --threshold 0.8
```

---

### 6. Checkpoint ablation (optimality sweep)

```bash
# Evaluate each reward-threshold checkpoint with fixed τ from thresholds.json
for CKPT in 030 050 070 090; do
  uv run python door_key/eval.py --mode ppo --size 5 \
    --model-path "runs/door_key/checkpoints_s5/model_reward_${CKPT}" \
    --tag "ckpt_r${CKPT}" --wandb-group doorkey_checkpoints
  uv run python door_key/eval.py --mode ask --slm qwen3.5-2b --size 5 \
    --model-path "runs/door_key/checkpoints_s5/model_reward_${CKPT}" \
    --tag "ckpt_r${CKPT}" --wandb-group doorkey_checkpoints
done
```

---

## Results Directory

```
door_key/results/
  ppo_results_s5.json          # PPO summary
  ppo_episodes_s5.csv          # per-episode log
  slm_qwen35_2b_results_s5.json
  ask_qwen35_2b_results_s5.json
  thresholds.json              # best τ per size × model
  # ... same pattern for s6, s8, s16
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `mean_reward` | Avg episodic reward (0–1; success = 1 − 0.9·step/max_steps) |
| `mean_success` | Fraction of episodes where goal was reached |
| `IR` | Intervention Rate — fraction of steps where SLM was called |
| `OR` | Override Rate — fraction of steps where SLM changed PPO's action |
| `slm_valid_rate` | Fraction of SLM calls that returned a valid action |

---

## W&B

- Project: `ask-pomdp-v2`
- Groups: `doorkey` (main), `doorkey_checkpoints` (ablation)
- All runs logged automatically.

```bash
wandb login   # one-time setup
```
