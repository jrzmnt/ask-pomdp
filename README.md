# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory) · MiniGrid-DoorKey-8x8 (sequential subtasks)  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Results

Summaries from JSON in [`results/`](results/), [`higher_lower/results/`](higher_lower/results/), and [`door_key/results/`](door_key/results/) (typically **100** test episodes). Details and checklist: [`experiments.md`](experiments.md).

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

### DoorKey-8x8 — Main comparison

Sequential subtask environment (find key → unlock door → reach goal). Same prompt-engineering stack as FourRooms (`basic` / `enriched` / `stateful_min` / `stateful`, optional rationale) and Optuna-tuned τ for ASK. Numbers below come from [`door_key/results/`](door_key/results/) (100 test episodes).

| Agent | Model | Reward ↑ | Success ↑ | IR | OR |
|-------|-------|:--------:|:---------:|:--:|:--:|
| PPO | — | 0.869 | 0.89 | — | — |
| SLM-only | Qwen3.5-2B | 0.000 | 0.00 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.582 | 0.62 | 1.00 | — |
| ASK | Qwen3.5-2B | **0.907** | **0.93** | 0.03 | 0.02 |
| ASK | Qwen3.5-4B | 0.905 | 0.93 | 0.03 | 0.02 |

ASK improves over the full-PPO baseline on both reward and success while only querying the SLM on ≈3% of steps. Optuna-tuned thresholds: τ=1.53 (2B), τ=1.46 (4B). Note: 2B SLM-only collapses on size-8 maps with the default `basic` prompt; ASK still benefits from the few well-targeted 2B interventions on uncertain PPO steps.

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

### Prompt ablation — DoorKey

No tagged ablation JSONs yet; the SLM-only numbers above use the default `basic` prompt (2B fails; 4B at 0.582). Run `SIZE=8 bash scripts/run_dk_prompt_ablation.sh` for the full `basic` / `enriched` / `stateful_min` / `stateful` / `stateful + rationale` matrix on Qwen3.5-2B.

### PPO optimality ablation (checkpoint training target)

**FourRooms** — PPO eval + ASK (Optuna, 10 trials per ckpt on weak PPO):

| Ckpt target | PPO Reward | Success | ASK-2B Reward | Success | IR-2B | ASK-4B Reward | Success | IR-4B |
|:-----------:|:----------:|:-------:|:-------------:|:-------:|:-----:|:-------------:|:-------:|:-----:|
| 0.1 | 0.183 | 0.19 | 0.338 | 0.36 | 0.55 | 0.345 | 0.38 | 0.73 |
| 0.3 | 0.203 | 0.21 | 0.378 | 0.40 | 0.41 | 0.405 | 0.45 | 0.45 |
| 0.5 | 0.451 | 0.47 | 0.567 | 0.62 | 0.45 | 0.486 | 0.53 | 0.28 |
| 0.7 | 0.601 | 0.63 | — | — | — | — | — | — |
| Full | 0.504 | 0.53 | **0.642** | **0.70** | 0.21 | 0.621 | 0.69 | 0.25 |

**HigherLower** — weak PPO checkpoints (r010–r040) share PPO reward ≈ 0.473 / acc. ≈ 0.712; ASK often queries every step (IR ≈ 1.0) on the weakest policies:

| Ckpt target | ASK-2B Reward | Acc. | IR-2B | ASK-4B Reward | Acc. | IR-4B |
|:-----------:|:-------------:|:----:|:-----:|:-------------:|:----:|:-----:|
| 0.1–0.3 | 0.512 | 0.732 | 1.00 | 0.520 | 0.736 | 1.00 |
| 0.4 | 0.523 | 0.737 | 0.83 | 0.520 | 0.736 | 1.00 |
| Full | 0.522 | 0.737 | 0.69 | 0.519 | 0.735 | 0.60 |

**DoorKey-8x8** — PPO eval per reward-threshold checkpoint; ASK row is the full-PPO result from the main table. Per-checkpoint ASK runs are launched by `door_key/scripts/eval_checkpoints.sh`:

| Ckpt target | PPO Reward | Success | ASK-2B Reward | Success | IR-2B | ASK-4B Reward | Success | IR-4B |
|:-----------:|:----------:|:-------:|:-------------:|:-------:|:-----:|:-------------:|:-------:|:-----:|
| 0.3 | 0.292 | 0.30 | _running_ | _running_ | — | _running_ | _running_ | — |
| 0.5 | 0.651 | 0.67 | _running_ | _running_ | — | _running_ | _running_ | — |
| 0.7 | 0.720 | 0.74 | _running_ | _running_ | — | _running_ | _running_ | — |
| Full | 0.869 | 0.89 | **0.907** | **0.93** | 0.03 | 0.905 | 0.93 | 0.03 |

---

## Setup

```bash
bash scripts/setup.sh
wandb login
```

## Running experiments

```bash
# FourRooms
bash scripts/run_fourrooms.sh
bash scripts/run_prompt_ablation.sh        # SLM-only prompt sweep
bash scripts/eval_checkpoints.sh           # PPO optimality ablation

# HigherLower
bash scripts/run_higherlower.sh
bash scripts/run_hl_prompt_ablation.sh

# DoorKey
bash door_key/scripts/train.sh                 # PPO + reward-threshold checkpoints
bash door_key/scripts/eval_ppo.sh   --size 8
bash door_key/scripts/eval_slm.sh   --size 8   --prompt-style stateful --prompt-rationale
bash door_key/scripts/eval_ask.sh   --size 8   --prompt-style stateful --prompt-rationale
SIZE=8 GPUS="0,1,2,3" bash door_key/scripts/eval_checkpoints.sh   # parallel ckpt sweep (1 ckpt / GPU)
SIZE=8 bash scripts/run_dk_prompt_ablation.sh                     # prompt ablation
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

door_key/
  env.py
  train.py
  eval.py
  scripts/
    train.sh
    eval_ppo.sh / eval_slm.sh / eval_ask.sh
    eval_checkpoints.sh        # parallel PPO+ASK across all reward-threshold checkpoints

scripts/
  run_fourrooms.sh / run_higherlower.sh
  run_prompt_ablation.sh / run_hl_prompt_ablation.sh / run_dk_prompt_ablation.sh
  eval_checkpoints.sh          # FourRooms PPO optimality ablation
```
