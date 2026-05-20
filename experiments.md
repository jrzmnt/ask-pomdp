# Experimentos — ASK-POMDP

**Ambientes:** `MiniGrid-FourRooms-v0` · `POPGym-HigherLower` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp-v2`

**Modelos SLM:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B

---

## FourRooms

- [x] Treinar PPO (com checkpoints a cada 500K) — `bash scripts/run_fourrooms.sh`
- [x] Avaliar PPO baseline → `results/ppo_results.json`
- [x] Avaliar SLM-only (2B, 4B) → `results/slm_qwen35_*_results.json`
- [x] Avaliar ASK (Optuna τ, 2B e 4B) → `results/ask_qwen35_2b_results.json`, `ask_qwen35_4b_results.json`
- [x] Ablation checkpoint (PPO optimality) → `ppo_results_ckpt_r*.json`, `ask_*_ckpt_r*.json` *(PPO: r010, r030, r050; ASK: r010, r030, r050 para 2B; r010, r030 para 4B)*
- [x] Ablation prompt (2B, 25 ep) → `results/slm_qwen35_2b_results_{basic,enriched,stateful_min,stateful,stateful_rat}.json`

## HigherLower

- [x] Treinar PPO + checkpoints reward → `runs/higher_lower/`
- [x] Avaliar PPO baseline → `higher_lower/results/ppo_results.json`
- [x] Avaliar SLM-only (2B, 4B) → `higher_lower/results/slm_qwen35_*_results.json`
- [x] Avaliar ASK (Optuna τ, 2B e 4B) → `higher_lower/results/ask_qwen35_*_results.json`
- [x] Ablation checkpoint → `ppo_results_ckpt_r010..r040.json`, `ask_*_ckpt_r010..r040.json`
- [ ] Ablation prompt HL (basic / enriched / stateful) → *(sem JSON com tag; produção usa stateful + rationale)*

---

## Resultados (JSON em `results/` e `higher_lower/results/`)

Valores dos ficheiros JSON no repositório (100 episódios de teste salvo indicação; ablação de prompt FR com **25** ep).

### FourRooms — PPO baseline

| Métrica | Valor |
|---------|-------|
| Ficheiro | `results/ppo_results.json` |
| `mean_reward` | 0.5040 |
| `success_rate` | 0.53 |
| `mean_length` | 249.46 |

### FourRooms — SLM-only (100 ep, stateful + rationale)

| Modelo | Ficheiro | `mean_reward` | `success_rate` | `mean_length` |
|--------|----------|---------------|----------------|---------------|
| Qwen3.5-2B | `slm_qwen35_2b_results.json` | 0.3031 | 0.34 | 350.48 |
| Qwen3.5-4B | `slm_qwen35_4b_results.json` | 0.5709 | 0.69 | 221.19 |

### FourRooms — ASK (modelo completo, Optuna τ)

| Modelo | Ficheiro | `τ` | `mean_reward` | `success_rate` | `mean_length` | `IR_mean` | `OR_mean` |
|--------|----------|-----|---------------|----------------|---------------|-----------|-----------|
| Qwen3.5-2B | `ask_qwen35_2b_results.json` | 0.4803 | **0.6420** | **0.70** | 182.20 | 0.205 | 0.018 |
| Qwen3.5-4B | `ask_qwen35_4b_results.json` | 0.4607 | 0.6211 | 0.69 | 193.27 | 0.251 | 0.018 |

ASK supera PPO (0.504) e SLM-only em reward e success nos dois tamanhos de modelo.

### FourRooms — Ablação checkpoint (reward alvo do ckpt PPO)

| Ckpt alvo | PPO `mean_reward` | PPO `success` | ASK-2B reward | ASK-2B success | IR-2B | ASK-4B reward | ASK-4B success | IR-4B |
|-----------|-------------------|---------------|---------------|----------------|-------|---------------|----------------|-------|
| 0.1 | 0.1833 | 0.19 | 0.3383 | 0.36 | 0.553 | 0.3455 | 0.38 | 0.734 |
| 0.3 | 0.2033 | 0.21 | 0.3776 | 0.40 | 0.411 | 0.4049 | 0.45 | 0.455 |
| 0.5 | 0.4508 | 0.47 | 0.5669 | 0.62 | 0.452 | — | — | — |
| Full | 0.5040 | 0.53 | 0.6420 | 0.70 | 0.205 | 0.6211 | 0.69 | 0.251 |

Ficheiros: `ppo_results_ckpt_r010|r030|r050.json`; `ask_qwen35_2b_results_ckpt_r010|r030|r050.json`; `ask_qwen35_4b_results_ckpt_r010|r030.json` (4B em r050 ainda em falta).

### FourRooms — Ablação de prompt (SLM-only, Qwen3.5-2B, 25 ep)

| Tag | `prompt_style` | Rationale | `mean_reward` | `success_rate` | `mean_length` |
|-----|----------------|-----------|---------------|----------------|---------------|
| `*_basic` | basic | no | 0.1958 | 0.20 | 402.32 |
| `*_enriched` | enriched | no | 0.0000 | 0.00 | 500.00 |
| `*_stateful_min` | stateful_min | no | 0.0752 | 0.08 | 462.64 |
| `*_stateful` | stateful | no | 0.0691 | 0.08 | 466.04 |
| `*_stateful_rat` | stateful | yes | 0.2689 | 0.32 | 368.40 |
| *(100 ep)* | stateful | yes | 0.3031 | 0.34 | 350.48 |

### HigherLower — PPO baseline

| Ficheiro | `mean_reward` | `mean_accuracy` |
|----------|---------------|-----------------|
| `higher_lower/results/ppo_results.json` | 0.4952 | 0.7231 |

### HigherLower — SLM-only (100 ep, stateful + rationale)

| Modelo | `mean_reward` | `mean_accuracy` |
|--------|---------------|-----------------|
| Qwen3.5-2B | 0.5125 | 0.7320 |
| Qwen3.5-4B | 0.5252 | 0.7384 |

### HigherLower — ASK (modelo completo)

| Modelo | `τ` | `mean_reward` | `mean_accuracy` | `IR_mean` | `OR_mean` |
|--------|-----|---------------|-----------------|-----------|-----------|
| Qwen3.5-2B | 0.0130 | 0.5221 | 0.7369 | 0.690 | 0.065 |
| Qwen3.5-4B | 0.0154 | 0.5190 | 0.7353 | 0.599 | 0.053 |

### HigherLower — Ablação checkpoint (PPO reward alvo 0.1–0.4)

PPO nas checkpoints r010–r040 partilha as mesmas métricas (reward ≈ 0.473, accuracy ≈ 0.712); ASK varia sobretudo em `τ` e `IR`.

| Ckpt alvo | ASK-2B reward | ASK-2B acc. | IR-2B | OR-2B | ASK-4B reward | ASK-4B acc. | IR-4B |
|-----------|---------------|-------------|-------|-------|---------------|-------------|-------|
| 0.1 | 0.5121 | 0.732 | 1.000 | 0.128 | 0.5198 | 0.736 | 1.000 |
| 0.2 | 0.5121 | 0.732 | 1.000 | 0.128 | 0.5198 | 0.736 | 1.000 |
| 0.3 | 0.5121 | 0.732 | 1.000 | 0.128 | 0.5198 | 0.736 | 1.000 |
| 0.4 | 0.5229 | 0.737 | 0.832 | 0.122 | 0.5198 | 0.736 | 1.000 |
| Full | 0.5221 | 0.737 | 0.690 | 0.065 | 0.5190 | 0.735 | 0.599 |

### Ainda em falta

- `results/ask_qwen35_4b_results_ckpt_r050.json` (ASK FourRooms 4B em ckpt r050)
- `results/ppo_results_ckpt_r020.json` e ckpts intermédios FR além de r010/r030/r050
- Ablação de prompt HL com tags em `higher_lower/results/`

## Pendente (ambos)

- [ ] Gerar tabelas e figuras para o paper (`plot_results.py`)
- [ ] Escrever paper no template CEUR-WS (6–9p)
- [ ] Submeter — https://openreview.net/group?id=ijcai.org/IJCAI-ECAI/2026/Workshop/PRL

---

## Configuração

**Métricas:** Reward, Accuracy (HL), Success / Episode Length (FR), IR, OR

**Thresholds:** Optuna (15 trials main, 10 trials ckpt) em seeds 0–99; teste em seeds 100–199

**W&B groups:** `fourrooms` · `fourrooms_checkpoints` · `fourrooms_prompt_ablation` · `higherlower` · `higherlower_checkpoints`

```bash
bash scripts/run_fourrooms.sh
bash scripts/run_higherlower.sh
bash scripts/run_prompt_ablation.sh
bash scripts/run_hl_prompt_ablation.sh
```
