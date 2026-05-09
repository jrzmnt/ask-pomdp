# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environment:** MiniGrid-FourRooms-v0 | **Observation:** 7×7 egocentric (147-dim) | **Actions:** TURN_LEFT, TURN_RIGHT, FORWARD

---

## Results

All agents evaluated on the same 200 test episodes (seeds 100–299). Threshold τ selected via Optuna on 100 validation episodes (seeds 0–99).

### Main comparison

| Agent | Model | Reward ↑ | Success Rate ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:--------------:|:------------:|:--:|:--:|
| PPO | — | 0.87 ± 0.24 | 93% | 69.12 ± 120.41 | — | — |
| SLM-only | Qwen2.5-0.5B | 0.04 ± 0.17 | 4% | 482.56 ± 85.94 | 1.00 | — |
| SLM-only | Qwen2.5-1.5B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| ASK | Qwen2.5-0.5B | — | — | — | — | — |
| ASK | Qwen2.5-1.5B | — | — | — | — | — |

*Reward and Episode Length reported as mean ± std. IR = Intervention Rate, OR = Overwrite Rate.*
*OR is not defined for SLM-only (no PPO reference). PPO mean episode length on successful episodes only: 36.69 steps.*

### Ablation: threshold τ (ASK + Qwen2.5-0.5B)

| τ | Reward ↑ | Success Rate ↑ | IR | OR |
|---|:--------:|:--------------:|:--:|:--:|
| 0.1 | — | — | — | — |
| 0.3 | — | — | — | — |
| 0.5 | — | — | — | — |
| 0.7 | — | — | — | — |
| 1.0 | — | — | — | — |
| 1.5 | — | — | — | — |
| 2.0 | — | — | — | — |

### Ablation: MC samples N (ASK + Qwen2.5-0.5B, τ = Optuna best)

| N | Reward ↑ | Success Rate ↑ | IR | OR |
|---|:--------:|:--------------:|:--:|:--:|
| 5 | — | — | — | — |
| 10 | — | — | — | — |
| 20 | — | — | — | — |
| 30 | — | — | — | — |
| 50 | — | — | — | — |

---

## Setup

```bash
bash scripts/setup.sh
wandb login
```

## Running experiments

```bash
bash scripts/train.sh               # ~3–5h  → runs/ppo/model.zip
bash scripts/eval_ppo.sh            # ~5min  → results/ppo_results.json
bash scripts/eval_slm.sh            # ~4–8h  → results/slm_qwen_*.json
bash scripts/eval_ask.sh            # ~4–8h  → results/ask_qwen_*.json
bash scripts/ablation_threshold.sh  # ~1h
bash scripts/ablation_mc_samples.sh # ~30min
bash scripts/ablation_always_ask.sh # ~1h
```

All runs are tracked in W&B project `ask-pomdp`.

## Smoke test

```bash
bash scripts/smoke_test.sh   # 14 tests, ~2–3 min, no GPU required
```

## Project structure

```
src/ask/
  envs/fourrooms.py       # MiniGrid wrapper (147-dim obs, ASCII view)
  uncertainty/entropy.py  # MC Dropout uncertainty estimation
  experiments/controls.py # EntropyGate (threshold gating)
  slm/model.py            # HuggingFace SLM wrapper (Qwen family)
  utils/ppo.py            # DropoutActorCriticPolicy
train_ppo.py              # PPO training
eval_ppo_slm.py           # Evaluation: PPO / SLM / ASK + Optuna
configs/rl/               # YAML configs
scripts/                  # Shell scripts for each pipeline stage
```
