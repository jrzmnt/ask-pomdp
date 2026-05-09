# Methodology — ASK-POMDP

## Environment

**MiniGrid-FourRooms-v0**

- 19×19 grid with four rooms connected by doorways
- Agent receives a 7×7 egocentric partial observation (147-dim flat vector after normalization)
- Action space: TURN_LEFT (0), TURN_RIGHT (1), FORWARD (2)
- Episode ends when the agent reaches the goal or hits `max_steps=500`
- Reward: +1 on goal, 0 otherwise (sparse)
- Seed controls agent start position and goal position; map layout is fixed

---

## Episode Splits

All splits are non-overlapping and defined solely by the random seed passed to `env.reset(seed=s)`.

| Split      | Seeds   | Size | Purpose                             |
|------------|---------|------|-------------------------------------|
| Validation | 0–99    | 100  | Optuna threshold search (τ) for ASK |
| Test       | 100–299 | 200  | Final evaluation of **all** agents  |

PPO, SLM-only and ASK are all evaluated on the same test seeds (100–299), ensuring a fair comparison. The Optuna validation set (0–99) is never used for final reporting.

Training uses SB3's internal environment seeding (non-deterministic across episodes), which is disjoint from both evaluation splits by design.

---

## Agents

### PPO (baseline)
- DropoutActorCriticPolicy, net_arch [256, 256], dropout_rate 0.2
- Trained for 5M steps on MiniGrid-FourRooms-v0 (`eval/mean_reward=0.81`, `success_rate=93%`)
- At test time: deterministic action, dropout **disabled** (`set_training_mode(False)`)

### SLM-only (baseline)
- Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B
- Off-the-shelf, no fine-tuning, thinking mode disabled for Qwen3 (`enable_thinking=False`)
- At each step: prompt with 7×7 ASCII view + facing direction → parse one-word action
- No PPO suggestion included (pure SLM decision)
- Invalid output → fallback to FORWARD

### ASK (main method)
- PPO + SLM gated by MC Dropout uncertainty
- At each step:
  1. Compute PPO action (deterministic)
  2. Estimate total entropy via N MC forward passes (dropout **enabled** during passes)
  3. If `total_entropy ≥ τ` → query SLM with PPO suggestion in prompt
  4. If SLM returns a valid action, use it; otherwise keep PPO action
- τ selected via Optuna on the validation split; final eval on the test split
- Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, Qwen3-1.7B

---

## Prompt Design

**SLM-only:** receives the 7×7 ASCII egocentric grid and facing direction. No PPO suggestion — the SLM acts as the sole decision maker.

**ASK:** same prompt plus `PPO autopilot suggests: <ACTION>`. The SLM acts as a consultant that can confirm or override the PPO.

---

## Uncertainty Estimation (MC Dropout)

```
total_entropy   = H(E_θ[p(a|o)])   — entropy of mean action distribution
aleatoric       = E_θ[H(p(a|o))]   — expected entropy under each sample
epistemic       = total - aleatoric — BALD approximation
```

- N=30 MC samples by default (ablation varies this)
- MLP extractor set to `train()` mode during passes to activate dropout; restored after

---

## Threshold Selection (Optuna)

- 15 trials, search space τ ∈ [0.1, 2.0] (uniform, TPE sampler)
- Objective: maximize mean reward on the validation split (seeds 0–99)
- Storage: `sqlite:///optuna.db` (resumable across runs with `load_if_exists=True`)
- Best τ saved to `results/thresholds.json`, W&B run config, and results JSON
- To reuse a threshold: `--threshold <value>` skips Optuna entirely

---

## Metrics

Per-episode schema is identical across all agents (PPO fields default to 0):

| Metric | Formula | PPO | SLM-only | ASK |
|--------|---------|:---:|:--------:|:---:|
| Reward | episode return | ✓ | ✓ | ✓ |
| Success Rate | reward > 0 | ✓ | ✓ | ✓ |
| Episode Length | steps until done | ✓ | ✓ | ✓ |
| Mean Length (success) | mean steps on successful eps | ✓ | ✓ | ✓ |
| IR | slm\_called / steps | 0 | 1.0 | computed |
| OR | slm\_overwrites / steps | 0 | nan | computed |
| slm\_valid\_rate | valid\_outputs / slm\_called | 0 | 1−invalid | computed |
| invalid\_action\_rate | invalid\_outputs / slm\_called | 0 | computed | computed |
| episode\_time\_s | wall clock per episode | ✓ | ✓ | ✓ |
| seed | env seed used | ✓ | ✓ | ✓ |

All aggregated metrics reported as mean ± std over 200 test episodes.

---

## Preliminary Results

| Agent | Model | Reward ↑ | Success ↑ | IR | OR |
|-------|-------|:--------:|:---------:|:--:|:--:|
| PPO | — | 0.87 ± 0.24 | 93% | — | — |
| SLM-only | Qwen2.5-0.5B | 0.04 ± 0.17 | 4% | 1.00 | — |
| SLM-only | Qwen2.5-1.5B | 0.00 ± 0.00 | 0% | 1.00 | — |
| SLM-only | Qwen3-0.6B | 0.00 ± 0.00 | 0% | 1.00 | — |
| SLM-only | Qwen3-1.7B | 0.00 ± 0.00 | 0% | 1.00 | — |
| ASK τ=0.55 | Qwen2.5-0.5B | 0.84 ± 0.29 | 89.5% | 0.14 | 0.10 |
| ASK τ=0.92 | Qwen2.5-1.5B | 0.87 ± 0.24 | 93% | 0.03 | 0.00 |
| ASK | Qwen3-0.6B | — | — | — | — |
| ASK | Qwen3-1.7B | — | — | — | — |

---

## Ablations

### 1. Threshold τ (`ablation_threshold.sh`)
- Fix τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0} (skips Optuna)
- Representative models: Qwen2.5-1.5B and Qwen3-1.7B
- Measures how IR, OR and reward change as τ varies
- **Expected:** low τ → high IR/OR, hurts performance (SLM is bad); high τ → IR→0, reward→PPO
- W&B group: `ablation_threshold`

### 2. MC Samples N (`ablation_mc_samples.sh`)
- Fix τ = Optuna best (from `results/thresholds.json`); vary N ∈ {5, 10, 20, 30, 50}
- Representative models: Qwen2.5-1.5B and Qwen3-1.7B
- Measures effect on reward and wall-clock time
- **Expected:** N≥20 should converge; lower N trades uncertainty quality for speed
- W&B group: `ablation_mc_samples`

### 3. Always-Ask (`ablation_always_ask.sh`)
- τ = 0 → SLM queried every step (IR = 1.0 by construction)
- All four models: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B
- Upper bound on SLM influence; shows that indiscriminate querying is suboptimal
- **Expected:** always-ask ≈ SLM-only performance (bad), confirming gating is necessary
- W&B group: `ablation_always_ask`

---

## Reproducibility

- Global seed: 42 (Python, NumPy, PyTorch, SB3)
- All eval seeds fixed and logged in W&B run config and per-episode CSV
- Model checkpoint: `runs/ppo/model.zip` and `runs/ppo/best_model/`
- Per-episode logs: `results/*.csv`; summaries: `results/*.json`
- Optuna studies: `optuna.db` (resumable); best thresholds: `results/thresholds.json`

---

## W&B Organization

| Group | job_type | Runs |
|-------|----------|------|
| `main` | `eval_ppo` | PPO baseline |
| `main` | `eval_slm` | SLM-only (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B) |
| `main` | `eval_ask` | ASK main result (all four models) |
| `ablation_threshold` | `ablation` | τ sweep (Qwen2.5-1.5B, Qwen3-1.7B) |
| `ablation_mc_samples` | `ablation` | N sweep (Qwen2.5-1.5B, Qwen3-1.7B) |
| `ablation_always_ask` | `ablation` | τ=0 (all four models) |

Each run logs: summary metrics to `run.summary`, per-episode `wandb.Table`, full config, and Optuna trial history (ASK only).
