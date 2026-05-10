# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory)  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Results

*Pending — run `bash scripts/run_fourrooms.sh` and `bash scripts/run_higherlower.sh`.*

### FourRooms — Main comparison

| Agent | Model | Reward ↑ | Success ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:---------:|:------------:|:--:|:--:|
| PPO | — | — | — | — | — | — |
| SLM-only | Qwen3.5-2B | — | — | — | 1.00 | — |
| SLM-only | Qwen3.5-4B | — | — | — | 1.00 | — |
| ASK | Qwen3.5-2B | — | — | — | — | — |
| ASK | Qwen3.5-4B | — | — | — | — | — |

### HigherLower — Main comparison

| Agent | Model | Reward ↑ | Accuracy ↑ | IR | OR |
|-------|-------|:--------:|:----------:|:--:|:--:|
| PPO | — | — | — | — | — |
| SLM-only | Qwen3.5-2B | — | — | 1.00 | — |
| SLM-only | Qwen3.5-4B | — | — | 1.00 | — |
| ASK | Qwen3.5-2B | — | — | — | — |
| ASK | Qwen3.5-4B | — | — | — | — |

### PPO Optimality Ablation

Effect of training quality on ASK intervention rate and reward.  
*Pending checkpoint ablation — see W&B group `fourrooms_checkpoints` / `higherlower_checkpoints`.*

| Checkpoint | Env | PPO Reward | ASK-2B Reward | IR-2B | ASK-4B Reward | IR-4B |
|------------|-----|:----------:|:-------------:|:-----:|:-------------:|:-----:|
| 10K / 500K steps | HL / FR | — | — | — | — | — |
| … | … | … | … | … | … | … |

---

## Setup

```bash
bash scripts/setup.sh
wandb login
```

## Running experiments

```bash
# Full pipelines (train + eval + checkpoint ablation)
bash scripts/run_fourrooms.sh     # ~13–15h on RTX 3060
bash scripts/run_higherlower.sh   # ~6h on RTX 3060

# Or step by step:
bash scripts/train.sh
bash scripts/eval_ppo.sh
bash scripts/eval_slm.sh
bash scripts/eval_ask.sh
bash scripts/eval_checkpoints.sh
```

## Project structure

```
src/ask/
  envs/fourrooms.py       # MiniGrid wrapper (147-dim obs, ASCII view)
  uncertainty/entropy.py  # MC Dropout uncertainty estimation
  slm/model.py            # HuggingFace SLM wrapper (Qwen family)
  utils/ppo.py            # DropoutActorCriticPolicy (MC Dropout)
train_ppo.py              # PPO training (FourRooms)
eval_ppo_slm.py           # Eval: PPO / SLM / ASK + Optuna (FourRooms)
higher_lower/
  env.py                  # POPGym HigherLower wrapper
  train.py                # PPO training with staged checkpoints
  eval.py                 # Eval: PPO / SLM / ASK + Optuna
scripts/
  run_fourrooms.sh        # Full FourRooms pipeline
  run_higherlower.sh      # Full HigherLower pipeline
```
