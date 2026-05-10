# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory)

---

## Results

All agents evaluated on 200 test episodes (seeds 100–299). Threshold τ tuned via Optuna on 100 validation episodes (seeds 0–99).

### FourRooms — Main comparison

| Agent | Model | Reward ↑ | Success Rate ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:--------------:|:------------:|:--:|:--:|
| PPO | — | 0.87 ± 0.24 | 93% | 69.12 ± 120.41 | — | — |
| SLM-only | Qwen2.5-0.5B | 0.04 ± 0.17 | 4% | 482.56 ± 85.94 | 1.00 | — |
| SLM-only | Qwen2.5-1.5B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| SLM-only | Qwen3-0.6B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| SLM-only | Qwen3-1.7B | 0.00 ± 0.00 | 0% | 500.00 ± 0.00 | 1.00 | — |
| ASK (τ=0.55) | Qwen2.5-0.5B | 0.84 ± 0.29 | 89.5% | 85.72 ± 143.78 | 0.14 | 0.10 |
| ASK (τ=0.92) | Qwen2.5-1.5B | 0.87 ± 0.24 | 93.0% | 69.12 ± 120.41 | 0.03 | 0.00 |
| ASK (τ=1.00) | Qwen3-0.6B | 0.83 ± 0.29 | 90.0% | 87.61 ± 144.77 | 0.01 | 0.01 |
| ASK (τ=1.48) | Qwen3-1.7B | 0.87 ± 0.24 | 93.0% | 69.12 ± 120.41 | 0.00 | 0.00 |

*Reward and Episode Length reported as mean ± std. IR = Intervention Rate, OR = Overwrite Rate.*
*OR not defined for SLM-only (no PPO reference). PPO mean episode length on successful episodes: 36.69 steps.*
*τ selected via Optuna on 100 validation episodes (seeds 0–99); all agents tested on seeds 100–299.*
*ASK Qwen3-1.7B: τ=1.48 ≈ max entropy for 3 actions (log₂3≈1.585) — SLM never queried; effectively pure PPO.*

### HigherLower — Main comparison

Episode = 51 steps (full deck). Reward ∈ [−1, 1] (±1/52 per correct/incorrect guess). Accuracy = % correct guesses. Random baseline: 50% accuracy.

| Agent | Model | Reward ↑ | Accuracy ↑ | IR | OR |
|-------|-------|:--------:|:----------:|:--:|:--:|
| PPO | — | 0.492 ± 0.102 | 72.0% ± 5.9% | — | — |
| SLM-only | Qwen2.5-0.5B | 0.007 ± 0.079 | 47.3% ± 4.4% | 1.00 | — |
| SLM-only | Qwen2.5-1.5B | 0.007 ± 0.079 | 47.3% ± 4.4% | 1.00 | — |
| SLM-only | Qwen3-0.6B | 0.153 ± 0.120 | 54.8% ± 6.3% | 1.00 | — |
| SLM-only | Qwen3-1.7B | 0.060 ± 0.118 | 50.0% ± 6.1% | 1.00 | — |
| ASK (τ=0.71) | Qwen2.5-0.5B | 0.492 ± 0.102 | 72.0% ± 5.9% | 0.0% | 0.0% |
| ASK (τ=0.08) | Qwen2.5-1.5B | 0.492 ± 0.102 | 72.0% ± 5.9% | 7.8% | 0.0% |
| ASK (τ=0.75) | Qwen3-0.6B   | 0.492 ± 0.102 | 72.0% ± 5.9% | 0.0% | 0.0% |
| ASK (τ=0.65) | Qwen3-1.7B   | 0.492 ± 0.103 | 72.0% ± 5.9% | 0.4% | 0.1% |

*Seeds: validation 0–99, test 100–299. All ASK agents match PPO reward — Optuna sets τ high enough that the SLM is rarely/never called, since PPO already plays near-optimally via card counting.*

### FourRooms — Ablation: threshold τ (FourRooms)

| τ | Reward (1.5B) ↑ | Success (1.5B) ↑ | IR | OR | Reward (3-1.7B) ↑ | Success (3-1.7B) ↑ | IR | OR |
|---|:---------------:|:----------------:|:--:|:--:|:-----------------:|:------------------:|:--:|:--:|
| 0.1 | 0.869 ± 0.242 | 93% | 28.8% | 0.0% | 0.287 ± 0.441 | 30%  | 47.8% | 41.2% |
| 0.3 | 0.869 ± 0.242 | 93% | 14.7% | 0.0% | 0.582 ± 0.454 | 63%  | 26.6% | 20.9% |
| 0.5 | 0.869 ± 0.242 | 93% |  8.7% | 0.0% | 0.578 ± 0.457 | 62.5%| 19.3% | 16.0% |
| 0.7 | 0.869 ± 0.242 | 93% |  6.5% | 0.0% | 0.672 ± 0.426 | 71.5%| 12.9% | 10.6% |
| 0.9 | 0.869 ± 0.242 | 93% |  3.8% | 0.0% | 0.769 ± 0.349 | 84%  |  6.3% |  5.0% |
| 1.0 | 0.869 ± 0.242 | 93% |  1.7% | 0.0% | 0.828 ± 0.301 | 88.5%|  2.3% |  1.4% |
| 1.2 | 0.869 ± 0.242 | 93% |  0.2% | 0.0% | 0.873 ± 0.234 | 93.5%|  0.2% |  0.1% |
| 1.4 | 0.869 ± 0.242 | 93% |  0.0% | 0.0% | 0.864 ± 0.249 | 92.5%|  0.1% |  0.1% |
| 1.6 | 0.869 ± 0.242 | 93% |  0.0% | 0.0% | 0.869 ± 0.242 | 93%  |  0.0% |  0.0% |
| 1.8 | 0.869 ± 0.242 | 93% |  0.0% | 0.0% | 0.869 ± 0.242 | 93%  |  0.0% |  0.0% |
| 2.0 | 0.869 ± 0.242 | 93% |  0.0% | 0.0% | 0.869 ± 0.242 | 93%  |  0.0% |  0.0% |

*Qwen2.5-1.5B: reward is flat across all τ (OR=0% — SLM always agrees with PPO). Qwen3-1.7B: low τ causes catastrophic performance degradation (OR>40% at τ=0.1), confirming the necessity of proper threshold tuning.*

### FourRooms — Ablation: MC samples N (τ = Optuna best)

| N | Reward (1.5B) ↑ | Success (1.5B) ↑ | IR | OR | Reward (3-1.7B) ↑ | Success (3-1.7B) ↑ | IR | OR |
|---|:---------------:|:----------------:|:--:|:--:|:-----------------:|:------------------:|:--:|:--:|
|  5 | 0.869 ± 0.242 | 93% | 3.2% | 0.0% | 0.869 ± 0.242 | 93% | 0.0% | 0.0% |
| 10 | 0.869 ± 0.242 | 93% | 3.2% | 0.0% | 0.869 ± 0.242 | 93% | 0.0% | 0.0% |
| 20 | 0.869 ± 0.242 | 93% | 3.4% | 0.0% | 0.869 ± 0.242 | 93% | 0.0% | 0.0% |
| 30 | 0.869 ± 0.242 | 93% | 3.4% | 0.0% | 0.869 ± 0.242 | 93% | 0.0% | 0.0% |
| 50 | 0.869 ± 0.242 | 93% | 3.4% | 0.0% | 0.869 ± 0.242 | 93% | 0.0% | 0.0% |

*Performance is robust to N across all values tested. IR varies minimally for Qwen2.5-1.5B (~3.2–3.4%); Qwen3-1.7B shows IR≈0% regardless of N due to its high Optuna-tuned τ.*

### FourRooms — Ablation: always-ask (τ = 0, IR = 100%)

| Model | Reward ↑ | Success ↑ | OR |
|-------|:--------:|:---------:|:--:|
| Qwen2.5-0.5B | 0.025 ± 0.154 |  2.5% | 96.5% |
| Qwen2.5-1.5B | 0.869 ± 0.242 | 93.0% |  0.0% |
| Qwen3-0.6B   | 0.000 ± 0.000 |  0.0% | 64.9% |
| Qwen3-1.7B   | 0.015 ± 0.119 |  1.5% | 95.2% |

*Qwen2.5-1.5B is the only model that matches PPO under always-ask (OR=0% — never overwrites), suggesting it has learned to navigate in-context. All other models degrade catastrophically when forced to act at every step.*

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
