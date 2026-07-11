metric                 = accuracy + mean_reasoning_tokens (joint; a "win" needs tokens down without accuracy collapsing)
direction              = lower tokens, accuracy within ~2pp of Round B's 85.71%
seed_experiments       = 2 (Experiment 0 control, Experiment 1 primary bet; 1b fallback only if needed)
seconds_per_experiment = Exp0 ~30min (150 steps); Exp1 ~2-3hr (mining + SFT + 300-step GRPO)
parallelism            = 1 (single shared GPU on eva02 — real parallelism is in research/mining, not concurrent GPU jobs)
compute_cap            = ~4 GPU-hours total for this round
max_generations        = 1 (single-pass matrix, not a multi-generation loop — GPU-bound sequential work)
--- spent ---
experiments_run = 0
gpu_min_used    = 0
best_metric     = Round B baseline: 85.71% acc / 58.3 mean tokens (ckpt_phase_eff3)
champion        = none yet (Round B remains champion until an Exp beats it on both axes)
