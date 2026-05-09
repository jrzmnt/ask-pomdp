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
| SLM-only | Qwen3-0.6B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| SLM-only | Qwen3-1.7B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| ASK (τ=0.55) | Qwen2.5-0.5B | 0.84 ± 0.29 | 89.5% | 85.72 ± 143.78 | 0.14 | 0.10 |
| ASK (τ=0.92) | Qwen2.5-1.5B | 0.87 ± 0.24 | 93% | 69.12 ± 120.41 | 0.03 | 0.00 |
| ASK (τ=?) | Qwen3-0.6B | — | — | — | — | — |
| ASK (τ=?) | Qwen3-1.7B | — | — | — | — | — |

*Reward and Episode Length reported as mean ± std. IR = Intervention Rate, OR = Overwrite Rate.*
*OR not defined for SLM-only (no PPO reference). PPO mean episode length on successful episodes: 36.69 steps.*
*τ selected via Optuna on 100 validation episodes (seeds 0–99); all agents tested on seeds 100–299.*

### Ablation: threshold τ

| τ | Reward ↑ | Success ↑ | IR | OR | Reward ↑ | Success ↑ | IR | OR |
|---|:--------:|:---------:|:--:|:--:|:--------:|:---------:|:--:|:--:|
| | **Qwen2.5-1.5B** | | | | **Qwen3-1.7B** | | | |
| 0.1 | — | — | — | — | — | — | — | — |
| 0.3 | — | — | — | — | — | — | — | — |
| 0.5 | — | — | — | — | — | — | — | — |
| 0.7 | — | — | — | — | — | — | — | — |
| 0.9 | — | — | — | — | — | — | — | — |
| 1.0 | — | — | — | — | — | — | — | — |
| 1.2 | — | — | — | — | — | — | — | — |
| 1.4 | — | — | — | — | — | — | — | — |
| 1.6 | — | — | — | — | — | — | — | — |
| 1.8 | — | — | — | — | — | — | — | — |
| 2.0 | — | — | — | — | — | — | — | — |

### Ablation: MC samples N (τ = Optuna best)

| N | Reward ↑ | Success ↑ | IR | OR | Reward ↑ | Success ↑ | IR | OR |
|---|:--------:|:---------:|:--:|:--:|:--------:|:---------:|:--:|:--:|
| | **Qwen2.5-1.5B** | | | | **Qwen3-1.7B** | | | |
| 5  | — | — | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — | — | — |
| 20 | — | — | — | — | — | — | — | — |
| 30 | — | — | — | — | — | — | — | — |
| 50 | — | — | — | — | — | — | — | — |

### Ablation: always-ask (τ = 0, IR = 100%)

| Agent | Model | Reward ↑ | Success ↑ | IR | OR |
|-------|-------|:--------:|:---------:|:--:|:--:|
| ASK τ=0 | Qwen2.5-0.5B | — | — | 1.00 | — |
| ASK τ=0 | Qwen2.5-1.5B | — | — | 1.00 | — |
| ASK τ=0 | Qwen3-0.6B | — | — | 1.00 | — |
| ASK τ=0 | Qwen3-1.7B | — | — | 1.00 | — |

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
