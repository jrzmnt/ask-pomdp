# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory)  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Results

Values below are taken from the JSON summaries in `results/` and `higher_lower/results/` (typically 100 test episodes). ASK rows stay empty until `ask_qwen35_*_results.json` exist.

### FourRooms — Main comparison

| Agent | Model | Reward ↑ | Success ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:---------:|:------------:|:--:|:--:|
| PPO | — | 0.504 | 0.53 | 249.5 | — | — |
| SLM-only | Qwen3.5-2B | 0.303 | 0.34 | 350.5 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.571 | 0.69 | 221.2 | 1.00 | — |
| ASK | Qwen3.5-2B | — | — | — | — | — |
| ASK | Qwen3.5-4B | — | — | — | — | — |

Sources: `results/ppo_results.json`; `results/slm_qwen35_2b_results.json` and `results/slm_qwen35_4b_results.json` (stateful prompt + rationale). An older SLM-only 2B run (`slm_qwen35_2b_results2.json`) scored 0 reward / 500 length and is omitted from the main table.

### HigherLower — Main comparison

| Agent | Model | Reward ↑ | Accuracy ↑ | IR | OR |
|-------|-------|:--------:|:----------:|:--:|:--:|
| PPO | — | 0.495 | 0.723 | — | — |
| SLM-only | Qwen3.5-2B | — | — | — | — |
| SLM-only | Qwen3.5-4B | — | — | — | — |
| ASK | Qwen3.5-2B | — | — | — | — |
| ASK | Qwen3.5-4B | — | — | — | — |

Source: `higher_lower/results/ppo_results.json`.

### PPO Optimality Ablation

Effect of training quality on ASK intervention rate and reward. Only a FourRooms PPO-at-low-return checkpoint summary is in-repo so far; ASK columns remain to be filled after checkpoint ASK runs.

| Checkpoint | Env | PPO Reward | ASK-2B Reward | IR-2B | ASK-4B Reward | IR-4B |
|------------|-----|:----------:|:-------------:|:-----:|:-------------:|:-----:|
| `ppo_results_ckpt_r010.json` (target mean return ≈ 0.1) | FourRooms | 0.183 | — | — | — | — |
| *(HigherLower + further FR ckpts)* | — | — | — | — | — | — |

See W&B groups `fourrooms_checkpoints` / `higherlower_checkpoints` for full sweeps.

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
