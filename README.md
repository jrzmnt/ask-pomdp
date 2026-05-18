# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory)  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Results

Summaries from JSON in [`results/`](results/) and [`higher_lower/results/`](higher_lower/results/) (typically **100** test episodes). Details and checklist: [`experiments.md`](experiments.md).

### FourRooms — Main comparison

| Agent | Model | Reward ↑ | Success ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:---------:|:------------:|:--:|:--:|
| PPO | — | 0.504 | 0.53 | 249.5 | — | — |
| SLM-only | Qwen3.5-2B | 0.303 | 0.34 | 350.5 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.571 | 0.69 | 221.2 | 1.00 | — |
| ASK | Qwen3.5-2B | **0.642** | **0.70** | **182.2** | 0.21 | 0.02 |
| ASK | Qwen3.5-4B | 0.621 | 0.69 | 193.3 | 0.25 | 0.02 |

ASK (full PPO) beats the PPO baseline on reward and success; SLM prompts use `stateful` + `prompt_rationale`.

### HigherLower — Main comparison

| Agent | Model | Reward ↑ | Accuracy ↑ | IR | OR |
|-------|-------|:--------:|:----------:|:--:|:--:|
| PPO | — | 0.495 | 0.723 | — | — |
| SLM-only | Qwen3.5-2B | 0.513 | 0.732 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.525 | 0.738 | 1.00 | — |
| ASK | Qwen3.5-2B | 0.522 | 0.737 | 0.69 | 0.06 |
| ASK | Qwen3.5-4B | 0.519 | 0.735 | 0.60 | 0.05 |

### Prompt ablation — FourRooms (SLM-only, Qwen3.5-2B, 25 ep)

| Prompt style | Rationale | Reward ↑ | Success ↑ | Ep. Length ↓ |
|--------------|-----------|:--------:|:---------:|:------------:|
| basic | no | 0.196 | 0.20 | 402.3 |
| enriched | no | 0.000 | 0.00 | 500.0 |
| stateful_min | no | 0.075 | 0.08 | 462.6 |
| stateful | no | 0.069 | 0.08 | 466.0 |
| stateful | yes | 0.269 | 0.32 | 368.4 |
| stateful + rationale *(100 ep)* | yes | **0.303** | **0.34** | **350.5** |

### Prompt ablation — HigherLower

No tagged ablation JSONs yet; production SLM/ASK use **stateful + rationale** (100 ep): SLM-2B 0.513 reward / 0.732 acc., SLM-4B 0.525 / 0.738. Run `bash scripts/run_hl_prompt_ablation.sh` for `basic` / `enriched` / `stateful` sweeps.

### PPO optimality ablation (checkpoint training target)

**FourRooms** — PPO eval + ASK (Optuna, 10 trials per ckpt on weak PPO):

| Ckpt target | PPO Reward | Success | ASK-2B Reward | Success | IR-2B | ASK-4B Reward | Success | IR-4B |
|:-----------:|:----------:|:-------:|:-------------:|:-------:|:-----:|:-------------:|:-------:|:-----:|
| 0.1 | 0.183 | 0.19 | 0.338 | 0.36 | 0.55 | 0.345 | 0.38 | 0.73 |
| 0.3 | 0.203 | 0.21 | 0.378 | 0.40 | 0.41 | 0.405 | 0.45 | 0.45 |
| 0.5 | 0.451 | 0.47 | 0.567 | 0.62 | 0.45 | — | — | — |
| Full | 0.504 | 0.53 | **0.642** | **0.70** | 0.21 | 0.621 | 0.69 | 0.25 |

**HigherLower** — weak PPO checkpoints (r010–r040) share PPO reward ≈ 0.473 / acc. ≈ 0.712; ASK often queries every step (IR ≈ 1.0) on the weakest policies:

| Ckpt target | ASK-2B Reward | Acc. | IR-2B | ASK-4B Reward | Acc. | IR-4B |
|:-----------:|:-------------:|:----:|:-----:|:-------------:|:----:|:-----:|
| 0.1–0.3 | 0.512 | 0.732 | 1.00 | 0.520 | 0.736 | 1.00 |
| 0.4 | 0.523 | 0.737 | 0.83 | 0.520 | 0.736 | 1.00 |
| Full | 0.522 | 0.737 | 0.69 | 0.519 | 0.735 | 0.60 |

---

## Setup

```bash
bash scripts/setup.sh
wandb login
```

## Running experiments

```bash
bash scripts/run_fourrooms.sh
bash scripts/run_higherlower.sh
bash scripts/run_prompt_ablation.sh
bash scripts/run_hl_prompt_ablation.sh
```

## Project structure

```
src/ask/
  envs/fourrooms.py
  uncertainty/entropy.py
  slm/model.py
  utils/ppo.py
train_ppo.py
eval_ppo_slm.py
higher_lower/
  env.py
  train.py
  eval.py
scripts/
  run_fourrooms.sh
  run_higherlower.sh
  run_prompt_ablation.sh
  run_hl_prompt_ablation.sh
```
