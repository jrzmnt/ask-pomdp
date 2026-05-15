# ASK-POMDP

**Uncertainty-Gated Language Assistance for Reinforcement Learning under Partial Observability**

> Short paper submitted to the PRL+CAIPI Workshop @ IJCAI-ECAI 2026.
> Extends [ASK (Monteiro et al., IJCNN 2026)](https://arxiv.org/abs/2604.02226) from fully observable to partially observable environments.

**Environments:** MiniGrid-FourRooms-v0 (POMDP navigation) · POPGym-HigherLower (card game with memory)  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Results

Summaries from JSON files in [`results/`](results/) and [`higher_lower/results/`](higher_lower/results/) (typically **100** test episodes; prompt ablations on FourRooms use **25** episodes). See [`experiments.md`](experiments.md) for the full checklist and per-file metrics.

### FourRooms — Main comparison

| Agent | Model | Reward ↑ | Success ↑ | Ep. Length ↓ | IR | OR |
|-------|-------|:--------:|:---------:|:------------:|:--:|:--:|
| PPO | — | 0.504 | 0.53 | 249.5 | — | — |
| SLM-only | Qwen3.5-2B | 0.303 | 0.34 | 350.5 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.571 | 0.69 | 221.2 | 1.00 | — |
| ASK | Qwen3.5-2B | **0.642** | **0.70** | **182.2** | 0.21 | 0.02 |
| ASK | Qwen3.5-4B | — | — | — | — | — |

ASK 2B uses Optuna τ ≈ 0.48 (`results/thresholds.json`). SLM rows use `prompt_style=stateful` and `prompt_rationale=true`. ASK 4B not yet in `results/`.

### HigherLower — Main comparison

| Agent | Model | Reward ↑ | Accuracy ↑ | IR | OR |
|-------|-------|:--------:|:----------:|:--:|:--:|
| PPO | — | 0.495 | 0.723 | — | — |
| SLM-only | Qwen3.5-2B | 0.513 | 0.732 | 1.00 | — |
| SLM-only | Qwen3.5-4B | 0.525 | 0.738 | 1.00 | — |
| ASK | Qwen3.5-2B | 0.522 | 0.737 | 0.69 | 0.06 |
| ASK | Qwen3.5-4B | 0.519 | 0.735 | 0.60 | 0.05 |

ASK/SLM prompts: `stateful` + `prompt_rationale` (see JSON metadata). PPO and ASK trained/evaluated on `runs/higher_lower/model`.

### Prompt ablation — FourRooms (SLM-only, Qwen3.5-2B)

25 episodes per condition (`scripts/run_prompt_ablation.sh`). Main 100-ep SLM run (stateful + rationale) shown for comparison.

| Prompt style | Rationale | Reward ↑ | Success ↑ | Ep. Length ↓ |
|--------------|-----------|:--------:|:---------:|:------------:|
| basic | no | 0.196 | 0.20 | 402.3 |
| enriched | no | 0.000 | 0.00 | 500.0 |
| stateful_min | no | 0.075 | 0.08 | 462.6 |
| stateful | no | 0.069 | 0.08 | 466.0 |
| stateful | yes | 0.269 | 0.32 | 368.4 |
| stateful + rationale *(100 ep)* | yes | **0.303** | **0.34** | **350.5** |

`enriched` alone timed out every episode at max length; **stateful + rationale** is the best 25-ep variant and matches the production SLM prompt for full evals.

### Prompt ablation — HigherLower (SLM-only)

No tagged ablation JSONs in `higher_lower/results/` yet. Production SLM/ASK runs use **stateful** prompt with **rationale** (100 ep):

| Agent | Model | Prompt | Reward ↑ | Accuracy ↑ |
|-------|-------|--------|:--------:|:----------:|
| SLM-only | Qwen3.5-2B | stateful + rationale | 0.513 | 0.732 |
| SLM-only | Qwen3.5-4B | stateful + rationale | 0.525 | 0.738 |

Run `bash scripts/run_hl_prompt_ablation.sh` to populate `basic` / `enriched` / `stateful` / `stateful_rat` rows (25 ep).

### PPO optimality ablation (checkpoint reward ≈ 0.1)

| Checkpoint | Env | PPO Reward | PPO Acc. / Success | ASK-2B Reward | ASK-2B Acc. / Success | IR-2B | OR-2B |
|------------|-----|:----------:|:------------------:|:-------------:|:---------------------:|:-----:|:-----:|
| Full model | FourRooms | 0.504 | 0.53 succ. | 0.642 | 0.70 succ. | 0.21 | 0.02 |
| `ckpt_r010` | FourRooms | 0.183 | 0.19 succ. | — | — | — | — |
| Full model | HigherLower | 0.495 | 0.723 acc. | 0.522 | 0.737 acc. | 0.69 | 0.06 |
| `ckpt_r010` | HigherLower | 0.473 | 0.712 acc. | 0.512 | 0.732 acc. | 1.00 | 0.13 |

FourRooms ASK on weak PPO checkpoint not yet run. HigherLower ASK-4B on ckpt: not in repo (only 2B ckpt ASK JSON present).

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

# Prompt ablations (25 ep, Qwen3.5-2B)
bash scripts/run_prompt_ablation.sh
bash scripts/run_hl_prompt_ablation.sh

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
  env.py                  # POPGym HigherLower wrapper + prompt memory
  train.py                # PPO training with staged checkpoints
  eval.py                 # Eval: PPO / SLM / ASK + Optuna
scripts/
  run_fourrooms.sh
  run_higherlower.sh
  run_prompt_ablation.sh
  run_hl_prompt_ablation.sh
```
