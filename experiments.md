# Experimentos — ASK-POMDP

**Ambientes:** `MiniGrid-FourRooms-v0` · `POPGym-HigherLower` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp-v2`

**Modelos SLM:** Qwen/Qwen3.5-2B · Qwen/Qwen3.5-4B

---

## FourRooms

- [ ] Treinar PPO (com checkpoints a cada 500K) — `bash scripts/run_fourrooms.sh`
- [ ] Avaliar PPO baseline → `results/ppo_results.json`
- [ ] Avaliar SLM-only (2B, 4B) → `results/slm_qwen35_*_results.json`
- [ ] Avaliar ASK (Optuna τ, 2B e 4B) → `results/ask_qwen35_*_results.json`
- [ ] Ablation checkpoint (PPO optimality) → `results/*_ckpt_*.json`

## HigherLower

- [ ] Treinar PPO (checkpoints: 10K, 25K, 50K, 100K, 200K, 500K) — `bash scripts/run_higherlower.sh`
- [ ] Avaliar PPO baseline → `higher_lower/results/ppo_results.json`
- [ ] Avaliar SLM-only (2B, 4B) → `higher_lower/results/slm_qwen35_*_results.json`
- [ ] Avaliar ASK (Optuna τ, 2B e 4B) → `higher_lower/results/ask_qwen35_*_results.json`
- [ ] Ablation checkpoint (PPO optimality) → `higher_lower/results/*_ckpt_*.json`

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
