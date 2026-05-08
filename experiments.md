# Experimentos — ASK-POMDP

**Ambiente:** `MiniGrid-FourRooms-v0` | **Deadline:** 15 mai 2026 | **W&B project:** `ask-pomdp`

---

- ✅ Setup — `bash scripts/setup.sh`
- ✅ Código implementado (`src/ask/`, `train_ppo.py`, `eval_ppo_slm.py`, `scripts/`)
- ✅ Smoke test — `bash scripts/smoke_test.sh` (14/14 passando)
- ✅ Login W&B — `wandb login` (necessário uma vez por máquina)
- [ ] Treinar PPO — `bash scripts/train.sh` (~3–5h) → `runs/ppo/model.zip`
- [ ] Avaliar PPO baseline — `bash scripts/eval_ppo.sh` (~5min) → `results/ppo_results.json`
- [ ] Avaliar SLM baseline — `bash scripts/eval_slm.sh` (~4–8h) → `results/slm_qwen_{0.5b,1.5b}_results.json`
- [ ] Avaliar ASK + Optuna — `bash scripts/eval_ask.sh` (~4–8h) → `results/ask_qwen_{0.5b,1.5b}_results.json`
- [ ] Ablation τ — `bash scripts/ablation_threshold.sh` (~1h) → `results/ask_*_threshold_*.json`
- [ ] Ablation N MC — `bash scripts/ablation_mc_samples.sh` (~30min, requer ASK) → `results/ask_*_mc*.json`
- [ ] Ablation always-ask — `bash scripts/ablation_always_ask.sh` (~1h) → `results/ask_*_always_ask.json`
- [ ] Criar `plot_results.py` e gerar tabelas/figuras
- [ ] Baixar template CEUR-WS e escrever o paper (6–9p)
- [ ] Submeter — https://openreview.net/group?id=ijcai.org/IJCAI-ECAI/2026/Workshop/PRL

---

**Métricas:** Reward, Episode Length, IR = `slm_called/steps`, OR = `slm_overwrites/steps`

**W&B runs:** cada etapa cria um run separado no projeto `ask-pomdp` (job types: `eval_ppo`, `eval_slm`, `eval_ask`)
