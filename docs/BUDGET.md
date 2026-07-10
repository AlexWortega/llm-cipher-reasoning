metric                = weighted composite (accuracy 0.5 + reasoning-cipher-adherence 0.35 + brevity 0.15), tracked alongside raw accuracy and raw adherence separately
direction             = higher
compute_cap           = single shared A6000, target ~4-6 GPU-hours for this round (shared box, be a good citizen)
# --- GRPO run config ---
base_model            = Qwen/Qwen3-4B-Instruct-2507
method                = LoRA (r=16, alpha=32) + GRPOTrainer (TRL 1.7.1), no vLLM (HF generate backend)
num_generations        = 8   # rollouts per prompt group
per_device_batch       = 4
grad_accum              = 4
max_completion_length  = 512
lr                      = 2e-6
beta (KL)               = 0.04   # nonzero per reward-hacking literature (DEEPRESEARCH.md)
curriculum              = Phase 0: correctness+format reward only, ~40 steps warmup
                          Phase 1: add cipher-adherence + brevity reward, remaining steps
max_steps               = 300 (Phase 0: 40, Phase 1: 260) -- reassess after Phase 1's first ~100
                          steps show a clear adherence trend or plateau
--- spent ---
steps_run       = 0
gpu_hours_used  = 0
best_adherence  = 0 (baseline: 0, model never trained on this before)
best_accuracy   = n/a (to be measured pre-training as baseline on the same held-out set)
champion        = none yet
