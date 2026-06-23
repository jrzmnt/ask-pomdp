# When to ASK under Partial Observability: Uncertainty-Gated Language Assistance for Reinforcement Learning

[![PRL+CAIPI @ IJCAI-ECAI 2026](https://img.shields.io/badge/PRL%2BCAIPI-IJCAI--ECAI%202026-blue)](https://prl-theworkshop.github.io/caipi_prl2026-ijcai/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Parent paper](https://img.shields.io/badge/ASK-IJCNN%202026-b31b1b)](https://arxiv.org/abs/2604.02226)

Official implementation extending **[ASK](https://arxiv.org/abs/2604.02226)** (Adaptive Safety through Knowledge) to partially observable reinforcement learning. ASK is an extrinsic method that improves generalization by selectively querying a Small Language Model (SLM) based on uncertainty estimates, without retraining the RL policy.

> ASK uses Monte Carlo Dropout to measure epistemic and aleatoric uncertainty at each step. When uncertainty exceeds a threshold τ, it queries an SLM for an action recommendation. In **MiniGrid-FourRooms** and **DoorKey-8×8**, ASK improves over the PPO baseline while querying the SLM on only a few percent of steps. In **POPGym HigherLower**, the gate learns to intervene only when the consultant carries useful information.

**Environments:** MiniGrid-FourRooms-v0 · POPGym-HigherLower · MiniGrid-DoorKey-8×8  
**SLM models:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B  
**W&B project:** `ask-pomdp-v2`

---

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- CUDA GPU recommended for SLM inference and PPO training

## Installation

```bash
uv sync
source .venv/bin/activate
```

Or use the setup script (creates the venv and installs dev dependencies):

```bash
bash scripts/setup.sh
source .venv/bin/activate
wandb login   # optional, for experiment logging
```

---

## Running Experiments

### FourRooms (main environment)

```bash
bash scripts/run_fourrooms.sh              # train PPO → eval SLM + ASK
bash scripts/run_prompt_ablation.sh        # SLM-only prompt sweep
bash scripts/eval_checkpoints.sh           # PPO optimality ablation
```

### HigherLower

```bash
bash scripts/run_higherlower.sh
bash scripts/run_hl_prompt_ablation.sh
```

### DoorKey

```bash
bash door_key/scripts/train.sh
bash door_key/scripts/eval_ppo.sh   --size 8
bash door_key/scripts/eval_slm.sh   --size 8   --prompt-style stateful --prompt-rationale
bash door_key/scripts/eval_ask.sh   --size 8   --prompt-style stateful --prompt-rationale
SIZE=8 GPUS="0,1,2,3" bash door_key/scripts/eval_checkpoints.sh
SIZE=8 bash scripts/run_dk_prompt_ablation.sh
```

### Full pipeline from scratch

```bash
bash scripts/run_all_from_scratch.sh
```

### Evaluation

```bash
# FourRooms
python eval_ppo_slm.py --mode ppo
python eval_ppo_slm.py --mode slm  --slm qwen3.5-2b
python eval_ppo_slm.py --mode ask  --slm qwen3.5-2b

# HigherLower
python higher_lower/eval.py --mode ppo
python higher_lower/eval.py --mode ask --slm qwen3.5-2b

# DoorKey
python door_key/eval.py --mode ppo --size 8
python door_key/eval.py --mode ask --slm qwen3.5-2b --size 8 --prompt-style stateful --prompt-rationale
```

Results are saved under `results/`, `higher_lower/results/`, and `door_key/results/`.

### Ablations and baselines

```bash                            # uniform-random SLM baseline
bash scripts/ablation_threshold_all.sh  # τ sweep
bash scripts/ablation_mc_samples_all.sh
bash scripts/ablation_prompt_ask_all.sh
```

---

## Configuration

| File | Description |
|---|---|
| `configs/rl/ppo.yaml` | PPO training config (environment, timesteps, network) |
| `configs/rl/ppo_smoke.yaml` | Short smoke-test training config |
| `prompts/` | SLM prompt templates (`basic`, `enriched`, `stateful`, …) |

**Key hyperparameters:**

- PPO training: 2×10⁶ timesteps (FourRooms / DoorKey); Stable-Baselines3 defaults otherwise
- MC Dropout: N=50 forward passes (default), dropout rate 0.2
- SLMs: Qwen family (0.5B–4B in this repo), off-the-shelf from HuggingFace, no fine-tuning
- Threshold τ: tuned per (environment, model) with Optuna (10 trials default)
- Evaluation: 100 episodes per configuration

Evaluation modes: `ppo` (baseline) · `slm` (SLM-only) · `ask` (uncertainty-gated ASK)

---

## Project Structure

```
├── configs/              # YAML experiment configs
├── eval_ppo_slm.py       # FourRooms: PPO / SLM / ASK evaluation
├── train_ppo.py          # FourRooms PPO training
├── prompts/              # SLM prompt templates
├── scripts/              # Training, eval, and ablation launchers
├── src/ask/
│   ├── envs/             # FourRooms environment wrapper
│   ├── experiments/      # Shared experiment controls
│   ├── slm/              # SLM loading, prompting, and parsing
│   ├── uncertainty/      # MC Dropout uncertainty estimation
│   └── utils/            # Callbacks, seeding, PPO helpers
├── higher_lower/         # POPGym HigherLower env + train/eval
└── door_key/             # MiniGrid DoorKey env + train/eval
```

---

## Acknowledgments

This work was partially supported by UK Research and Innovation [grant number EP/S023356/1], in the UKRI Centre for Doctoral Training in Safe and Trusted Artificial Intelligence (www.safeandtrustedai.org), and by the Kunumi Institute (https://www.kunuminst.org/), through individual grants awarded to the authors.

---

## Citation

If you use this code, please cite the workshop paper (link forthcoming) and the original ASK paper:

```bibtex
@inproceedings{monteiro2026askpomdp,
  title     = {When to ASK under Partial Observability: Uncertainty-Gated Language Assistance for Reinforcement Learning},
  author    = {Juarez Monteiro and Nathan Gavenski and Gianlucca Zuin and Adriano Veloso},
  booktitle = {Proceedings of the Joint Workshop on Planning for Complex Real-World Applications (CAIPI) and Bridging the Gap Between AI Planning and (Reinforcement) Learning (PRL) at IJCAI-ECAI 2026},
  year      = {2026},
  note      = {Workshop paper — link forthcoming},
}

@inproceedings{monteiro2026ask,
  title     = {When to ASK: Uncertainty-Gated Language Assistance for Reinforcement Learning},
  author    = {Juarez Monteiro and Nathan Gavenski and Gianlucca Zuin and Adriano Veloso},
  booktitle = {Proceedings of the 2026 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.02226},
}
```
