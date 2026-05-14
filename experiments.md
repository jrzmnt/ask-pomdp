# Experimentos — ASK-POMDP

**Ambientes:** `MiniGrid-FourRooms-v0` · `POPGym-HigherLower` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp-v2`

**Modelos SLM:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B

---

## FourRooms

- [x] Treinar PPO (com checkpoints a cada 500K) — `bash scripts/run_fourrooms.sh` *(métricas em `results/`; ver `runs/ppo/` para artefactos)*
- [x] Avaliar PPO baseline → `results/ppo_results.json`
- [x] Avaliar SLM-only (2B, 4B) → `results/slm_qwen35_*_results.json`
- [ ] Avaliar ASK (Optuna τ, 2B e 4B) → `results/ask_qwen35_*_results.json` *(sem JSON ainda)*
- [x] Ablation checkpoint (PPO optimality) → `results/ppo_results_ckpt_r010.json` *(um ficheiro; acrescentar mais `*_ckpt_*.json` se existirem)*

## HigherLower

- [ ] Treinar PPO (checkpoints: 10K, 25K, 50K, 100K, 200K, 500K) — `bash scripts/run_higherlower.sh`
- [x] Avaliar PPO baseline → `higher_lower/results/ppo_results.json`
- [ ] Avaliar SLM-only (2B, 4B) → `higher_lower/results/slm_qwen35_*_results.json`
- [ ] Avaliar ASK (Optuna τ, 2B e 4B) → `higher_lower/results/ask_qwen35_*_results.json`
- [ ] Ablation checkpoint (PPO optimality) → `higher_lower/results/*_ckpt_*.json`

---

## Resultados (JSON em `results/`)

Valores copiados dos ficheiros JSON atuais (seeds de teste conforme script de avaliação; tipicamente 100 episódios).

### FourRooms — PPO baseline

| Métrica | Valor |
|---------|-------|
| Ficheiro | `results/ppo_results.json` |
| `n_episodes` | 100 |
| `mean_reward` | 0.5040 |
| `std_reward` | 0.4758 |
| `success_rate` | 0.53 |
| `mean_length` | 249.46 |
| `std_length` | 236.71 |
| `mean_length_success` | 27.28 |

### FourRooms — PPO checkpoint (ablação)

| Métrica | Valor |
|---------|-------|
| Ficheiro | `results/ppo_results_ckpt_r010.json` |
| `checkpoint_reward` | 0.1 |
| `n_episodes` | 100 |
| `mean_reward` | 0.1833 |
| `std_reward` | 0.3787 |
| `success_rate` | 0.19 |
| `mean_length` | 408.70 |
| `std_length` | 188.62 |
| `mean_length_success` | 19.47 |

### FourRooms — SLM-only (Qwen3.5)

| Ficheiro | Modelo | Prompt / notas | `mean_reward` | `std_reward` | `success_rate` | `mean_length` | `mean_length_success` | `invalid_action_rate` |
|----------|--------|------------------|---------------|--------------|----------------|---------------|----------------------|-------------------------|
| `results/slm_qwen35_2b_results.json` | Qwen3.5-2B | `prompt_style`: stateful, `prompt_rationale`: true | 0.3031 | 0.4294 | 0.34 | 350.48 | 60.24 | 0.0 |
| `results/slm_qwen35_2b_results2.json` | Qwen3.5-2B | *(sem metadados de prompt no JSON; run histórico)* | 0.0 | 0.0 | 0.0 | 500.0 | — *(NaN)* | 0.0 |
| `results/slm_qwen35_4b_results.json` | Qwen3.5-4B | `prompt_style`: stateful, `prompt_rationale`: true | 0.5709 | 0.4220 | 0.69 | 221.19 | 95.93 | 0.0 |

Para o 2B em `slm_qwen35_2b_results2.json`, `mean_length_success` é `NaN` no ficheiro (nenhum sucesso).

### HigherLower — PPO baseline

| Métrica | Valor |
|---------|-------|
| Ficheiro | `higher_lower/results/ppo_results.json` |
| `n_episodes` | 100 |
| `mean_reward` | 0.4952 |
| `std_reward` | 0.1075 |
| `mean_accuracy` | 0.7231 |
| `std_accuracy` | 0.0622 |

### Ainda em falta (sem JSON no repositório)

- `results/ask_qwen35_2b_results.json`, `results/ask_qwen35_4b_results.json` (ASK FourRooms)
- `higher_lower/results/slm_*.json`, `higher_lower/results/ask_*.json` (HigherLower SLM/ASK)

## Pendente (ambos)

- [ ] Gerar tabelas e figuras para o paper (`plot_results.py`)
- [ ] Escrever paper no template CEUR-WS (6–9p)
- [ ] Submeter — https://openreview.net/group?id=ijcai.org/IJCAI-ECAI/2026/Workshop/PRL

---

## Configuração

**Métricas:** Reward, Accuracy (HL), Episode Length (FR), IR = `slm_called/steps`, OR = `slm_overwrites/steps`

**Thresholds:** selecionados via Optuna (15 trials) em seeds 0–99; avaliação final em seeds 100–299

**W&B groups:**
- `fourrooms` — avaliações principais FourRooms
- `fourrooms_checkpoints` — ablação de otimalidade PPO (FourRooms)
- `higherlower` — avaliações principais HigherLower
- `higherlower_checkpoints` — ablação de otimalidade PPO (HigherLower)

**Scripts completos:**
```bash
bash scripts/run_fourrooms.sh    # ~13–15h
bash scripts/run_higherlower.sh  # ~6h
```
