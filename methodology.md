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

| Split      | Seeds     | Size | Purpose                                       |
|------------|-----------|------|-----------------------------------------------|
| Validation | 0–99      | 100  | Optuna threshold search (τ) for ASK           |
| Test       | 100–299   | 200  | Final evaluation of **all** agents            |

**PPO, SLM-only and ASK are all evaluated on the same test seeds (100–299)**, ensuring a fair comparison. The Optuna validation set (0–99) is never used for final reporting.

Training uses SB3's internal environment seeding (non-deterministic across episodes), which is disjoint from both evaluation splits by design.

---

## Agents

### PPO (baseline)
- DropoutActorCriticPolicy, net_arch [256, 256], dropout_rate 0.2
- Trained for 5M steps on MiniGrid-FourRooms-v0
- At test time: deterministic action, dropout **disabled** (`set_training_mode(False)`)

### SLM-only (baseline)
- Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (off-the-shelf, no fine-tuning)
- At each step: build prompt from egocentric view + facing direction, parse one-word action
- Invalid output → fallback to FORWARD

### ASK (main method)
- PPO + SLM gated by MC Dropout uncertainty
- At each step:
  1. Compute PPO action (deterministic)
  2. Estimate total entropy via N MC forward passes (dropout **enabled** during passes)
  3. If total\_entropy ≥ τ → query SLM; if SLM returns a valid action, use it; otherwise keep PPO action
- τ selected via Optuna on the validation split; final eval on the test split

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

- 15 trials, search space τ ∈ [0.1, 2.0] (uniform)
- Objective: maximize mean reward on the validation split (seeds 0–99)
- Storage: `sqlite:///optuna.db` (resumable)
- Early stop if any trial achieves mean reward ≥ 0.999
- Best τ is stored in the W&B run config and in the results JSON

---

## Metrics

Computed per episode, then aggregated over the test split:

| Metric | Formula | Notes |
|--------|---------|-------|
| Reward | episode return | 1 = goal reached, 0 = timeout |
| Success Rate | fraction of episodes with reward > 0 | primary metric |
| Episode Length | steps until done | lower = more efficient |
| Mean Length (success) | mean steps on successful episodes | efficiency conditional on success |
| IR (Intervention Rate) | slm\_called / steps | fraction of steps where SLM was queried |
| OR (Overwrite Rate) | slm\_overwrites / steps | fraction of steps where SLM changed the action |
| Invalid Action Rate | invalid\_slm\_outputs / steps | SLM-only and ASK; lower = more reliable |

All aggregated metrics are reported as mean ± std over the 200 test episodes.

---

## Ablations

### 1. Threshold τ (`ablation_threshold.sh`)
- Fix τ ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0} (skips Optuna)
- Measure how IR, OR and reward change as τ varies
- **Expected:** low τ → high IR/OR, diminishing returns; high τ → IR→0, reward→PPO level
- W&B group: `ablation_threshold`

### 2. MC Samples N (`ablation_mc_samples.sh`)
- Fix τ = Optuna best; vary N ∈ {5, 10, 20, 30, 50}
- Measure effect on reward and wall-clock time per episode
- **Expected:** N≥20 should converge; lower N trades accuracy for speed
- W&B group: `ablation_mc_samples`

### 3. Always-Ask (`ablation_always_ask.sh`)
- τ = 0 → SLM queried every step (IR = 1.0 by construction)
- Upper bound on SLM influence; reveals whether more intervention helps or hurts
- **Expected:** if SLM alone < PPO, always-ask < Optuna-τ
- W&B group: `ablation_always_ask`

---

## Reproducibility

- Global seed: 42 (Python, NumPy, PyTorch, SB3)
- All eval seeds are fixed and logged in W&B run config
- Model checkpoint saved to `runs/ppo/model.zip` and `runs/ppo/best_model/`
- All per-episode logs saved to `results/*.csv`; summaries to `results/*.json`
- Optuna study persisted in `optuna.db`

---

## W&B Organization

| Group | job_type | Runs |
|-------|----------|------|
| `main` | `eval_ppo` | PPO baseline |
| `main` | `eval_slm` | SLM-only (0.5b, 1.5b) |
| `main` | `eval_ask` | ASK main result (0.5b, 1.5b) |
| `ablation_threshold` | `ablation` | τ sweep |
| `ablation_mc_samples` | `ablation` | N sweep |
| `ablation_always_ask` | `ablation` | τ=0 |

Each run logs: summary metrics to `run.summary`, per-episode `wandb.Table`, and full config.
