# Experimentos — ASK-POMDP

**Ambientes:** `MiniGrid-FourRooms-v0` · `POPGym-HigherLower` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp-v2`

**Modelos SLM:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B

---

## FourRooms

- [x] Treinar PPO (com checkpoints a cada 500K) — `bash scripts/run_fourrooms.sh`
- [x] Avaliar PPO baseline → `results/ppo_results.json`
- [x] Avaliar SLM-only (2B, 4B) → `results/slm_qwen35_*_results.json`
- [x] Avaliar ASK (Optuna τ, 2B) → `results/ask_qwen35_2b_results.json`
- [ ] Avaliar ASK (Optuna τ, 4B) → `results/ask_qwen35_4b_results.json` *(ainda em falta)*
- [x] Ablation checkpoint (PPO optimality) → `results/ppo_results_ckpt_r010.json` *(só PPO; ASK em ckpt pendente para FR)*
- [x] Ablation prompt (2B, 25 ep) → `results/slm_qwen35_2b_results_{basic,enriched,stateful_min,stateful,stateful_rat}.json`

## HigherLower

- [x] Treinar PPO + checkpoints reward → `runs/higher_lower/` *(script completo; ckpt r010 em `results/`)*
- [x] Avaliar PPO baseline → `higher_lower/results/ppo_results.json`
- [x] Avaliar SLM-only (2B, 4B) → `higher_lower/results/slm_qwen35_*_results.json`
- [x] Avaliar ASK (Optuna τ, 2B e 4B) → `higher_lower/results/ask_qwen35_*_results.json`
- [x] Ablation checkpoint (parcial) → `ppo_results_ckpt_r010.json`, `ask_qwen35_2b_results_ckpt_r010.json`
- [ ] Ablation prompt HL (basic / enriched / stateful) → *(sem JSON com tag; SLM principal usa `stateful` + `prompt_rationale`)*

---

## Resultados (JSON em `results/` e `higher_lower/results/`)

Valores copiados dos ficheiros JSON no repositório (tipicamente 100 episódios de teste; ablações de prompt com **25** episódios salvo indicação).

### FourRooms — PPO baseline

| Métrica | Valor |
|---------|-------|
| Ficheiro | `results/ppo_results.json` |
| `n_episodes` | 100 |
| `mean_reward` | 0.5040 |
| `std_reward` | 0.4758 |
| `success_rate` | 0.53 |
| `mean_length` | 249.46 |
| `mean_length_success` | 27.28 |

### FourRooms — PPO checkpoint (ablação otimalidade)

| Métrica | Valor |
|---------|-------|
| Ficheiro | `results/ppo_results_ckpt_r010.json` |
| `checkpoint_reward` | 0.1 |
| `mean_reward` | 0.1833 |
| `success_rate` | 0.19 |
| `mean_length` | 408.70 |

### FourRooms — SLM-only (100 ep, prompt principal)

| Ficheiro | Modelo | Prompt | `mean_reward` | `success_rate` | `mean_length` | `invalid_action_rate` |
|----------|--------|--------|---------------|----------------|---------------|------------------------|
| `results/slm_qwen35_2b_results.json` | Qwen3.5-2B | stateful + rationale | 0.3031 | 0.34 | 350.48 | 0.0 |
| `results/slm_qwen35_4b_results.json` | Qwen3.5-4B | stateful + rationale | 0.5709 | 0.69 | 221.19 | 0.0 |

Run histórico sem prompt enriquecido: `slm_qwen35_2b_results2.json` → reward 0.0, success 0.0, length 500 (omitido das tabelas principais).

### FourRooms — ASK (Optuna τ)

| Ficheiro | Modelo | `τ` | `mean_reward` | `success_rate` | `mean_length` | `IR_mean` | `OR_mean` | `slm_valid_rate` |
|----------|--------|-----|---------------|----------------|---------------|-----------|-----------|------------------|
| `results/ask_qwen35_2b_results.json` | Qwen3.5-2B | 0.4803 | 0.6420 | 0.70 | 182.20 | 0.205 | 0.018 | 0.83 |
| `results/ask_qwen35_4b_results.json` | Qwen3.5-4B | — | — | — | — | — | — | — |

Optuna (seeds 0–99): `results/thresholds.json` → `fourrooms_qwen35_2b`, eval_reward Optuna = 0.6355.

### FourRooms — Ablação de prompt (SLM-only, Qwen3.5-2B, 25 ep)

| Tag / ficheiro | `prompt_style` | `prompt_rationale` | `mean_reward` | `success_rate` | `mean_length` |
|----------------|----------------|------------------|---------------|----------------|---------------|
| `*_basic` | basic | false | 0.1958 | 0.20 | 402.32 |
| `*_enriched` | enriched | false | 0.0000 | 0.00 | 500.00 |
| `*_stateful_min` | stateful_min | false | 0.0752 | 0.08 | 462.64 |
| `*_stateful` | stateful | false | 0.0691 | 0.08 | 466.04 |
| `*_stateful_rat` | stateful | true | 0.2689 | 0.32 | 368.40 |

Melhor entre ablações de 25 ep: **stateful + rationale** (0.269 reward); run de 100 ep com o mesmo prompt: **0.303** (`slm_qwen35_2b_results.json`).

### HigherLower — PPO baseline

| Métrica | Valor |
|---------|-------|
| Ficheiro | `higher_lower/results/ppo_results.json` |
| `n_episodes` | 100 |
| `mean_reward` | 0.4952 |
| `mean_accuracy` | 0.7231 |

### HigherLower — PPO checkpoint (ckpt r010)

| Métrica | Valor |
|---------|-------|
| Ficheiro | `higher_lower/results/ppo_results_ckpt_r010.json` |
| `checkpoint_reward` | 0.1 |
| `mean_reward` | 0.4733 |
| `mean_accuracy` | 0.7120 |

### HigherLower — SLM-only (100 ep)

| Ficheiro | Modelo | Prompt | `mean_reward` | `mean_accuracy` |
|----------|--------|--------|---------------|-----------------|
| `higher_lower/results/slm_qwen35_2b_results.json` | Qwen3.5-2B | stateful + rationale | 0.5125 | 0.7320 |
| `higher_lower/results/slm_qwen35_4b_results.json` | Qwen3.5-4B | stateful + rationale | 0.5252 | 0.7384 |

### HigherLower — ASK (Optuna τ, prompt stateful + rationale)

| Ficheiro | Modelo | `τ` | `mean_reward` | `mean_accuracy` | `IR_mean` | `OR_mean` |
|----------|--------|-----|---------------|-----------------|-----------|-----------|
| `higher_lower/results/ask_qwen35_2b_results.json` | Qwen3.5-2B | 0.0130 | 0.5221 | 0.7369 | 0.690 | 0.065 |
| `higher_lower/results/ask_qwen35_4b_results.json` | Qwen3.5-4B | 0.0154 | 0.5190 | 0.7353 | 0.599 | 0.053 |
| `ask_qwen35_2b_results_ckpt_r010.json` | Qwen3.5-2B (PPO ckpt) | 0.4212 | 0.5121 | 0.7318 | 1.000 | 0.128 |

Thresholds em `higher_lower/results/thresholds.json`: `higherlower_qwen35_2b`, `higherlower_qwen35_4b`, `higherlower_qwen35_2b_ckpt_r010`.

### Ainda em falta

- `results/ask_qwen35_4b_results.json` (ASK FourRooms, 4B)
- Ablações de prompt HL com tags (`basic` / `enriched` / `stateful`) em `higher_lower/results/`
- ASK FourRooms / HigherLower em checkpoints FR adicionais (além de `ckpt_r010` HL)

## Pendente (ambos)

- [ ] Gerar tabelas e figuras para o paper (`plot_results.py`)
- [ ] Escrever paper no template CEUR-WS (6–9p)
- [ ] Submeter — https://openreview.net/group?id=ijcai.org/IJCAI-ECAI/2026/Workshop/PRL

---

## Configuração

**Métricas:** Reward, Accuracy (HL), Success / Episode Length (FR), IR = `slm_called/steps`, OR = `slm_overwrites/steps`

**Thresholds:** selecionados via Optuna (15 trials) em seeds 0–99; avaliação final em seeds 100–199 (`N_EVAL_EPISODES` offset)

**W&B groups:**
- `fourrooms` · `fourrooms_checkpoints` · `fourrooms_prompt_ablation`
- `higherlower` · `higherlower_checkpoints` · `higherlower_prompt_ablation`

**Scripts completos:**
```bash
bash scripts/run_fourrooms.sh
bash scripts/run_higherlower.sh
bash scripts/run_prompt_ablation.sh      # FourRooms prompt ablation (25 ep)
bash scripts/run_hl_prompt_ablation.sh  # HigherLower (quando corrido)
```
