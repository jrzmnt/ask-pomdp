# Experimentos — ASK-POMDP

**Ambientes:** `MiniGrid-FourRooms-v0` · `POPGym-HigherLower` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp`

---

- ✅ Setup — `bash scripts/setup.sh`
- ✅ Código implementado (`src/ask/`, `train_ppo.py`, `eval_ppo_slm.py`, `scripts/`)
- ✅ Smoke test — `bash scripts/smoke_test.sh` (14/14 passando)
- ✅ Login W&B — `wandb login` (necessário uma vez por máquina)
- ✅ Treinar PPO — `bash scripts/train.sh` (~3–5h) → `runs/ppo/model.zip`
- ✅ Avaliar PPO baseline — `bash scripts/eval_ppo.sh` (~5min) → `results/ppo_results.json`
- ✅ Avaliar SLM baseline Qwen2.5 — `bash scripts/eval_slm.sh` (~4–8h) → `results/slm_qwen25_{0.5b,1.5b}_results.json`
- ✅ Avaliar SLM baseline Qwen3 — `bash scripts/eval_slm_qwen3.sh` → `results/slm_qwen3_{0.6b,1.7b}_results.json`
- ✅ Avaliar ASK Qwen2.5 — `bash scripts/eval_ask.sh` → `results/ask_qwen25_{0.5b,1.5b}_results.json`
- ✅ Avaliar ASK Qwen3-0.6B → `results/ask_qwen3_0.6b_results.json` (τ=1.00, success=90%, IR=0.01)
- ✅ Avaliar ASK Qwen3-1.7B → `results/ask_qwen3_1.7b_results.json` (τ=1.48, IR=0%, success=93%)
- ✅ Ablation τ — `bash scripts/ablation_threshold.sh` → `results/ask_*_threshold_*.json`
- ✅ Ablation N MC — `bash scripts/ablation_mc_samples.sh` → `results/ask_*_mc*.json`
- ✅ Ablation always-ask — `bash scripts/ablation_always_ask.sh` → `results/ask_*_always_ask.json`

**HigherLower (POPGym)**

- ✅ Treinar PPO — `bash higher_lower/scripts/train.sh` → `runs/higher_lower/model.zip`
- ✅ Avaliar PPO baseline — `bash higher_lower/scripts/eval_ppo.sh` → `higher_lower/results/ppo_results.json`
- ✅ Avaliar SLM baseline — `bash higher_lower/scripts/eval_slm.sh` → `higher_lower/results/slm_*_results.json`
- ✅ Avaliar ASK — `bash higher_lower/scripts/eval_ask.sh` → `higher_lower/results/ask_*_results.json`

**Pendente (ambos os ambientes)**

- [ ] Criar `plot_results.py` e gerar tabelas/figuras
- [ ] Baixar template CEUR-WS e escrever o paper (6–9p)
- [ ] Submeter — https://openreview.net/group?id=ijcai.org/IJCAI-ECAI/2026/Workshop/PRL

---

**Modelos SLM:** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3.5-0.8B, Qwen3.5-2B

**Métricas:** Reward, Success Rate, Episode Length, IR = `slm_called/steps`, OR = `slm_overwrites/steps`

**Thresholds Optuna:** salvos em `results/thresholds.json` e `optuna.db`

**W&B runs:** cada etapa cria um run separado no projeto `ask-pomdp` (job types: `eval_ppo`, `eval_slm`, `eval_ask`, `ablation`)
