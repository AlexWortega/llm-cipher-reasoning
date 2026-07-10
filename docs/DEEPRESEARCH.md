## Direct precedent (closest possible match)
- **Motwani/Yale et al.(?) "LLMs Can Learn and Generalize Steganographic CoT Under Process
  Supervision"** (arXiv:2506.01926) — near-exact prior version of this experiment. PPO (not GRPO,
  but same RL-shapes-CoT-tokens family), reward = task_reward − monitor_penalty, Qwen2.5-3B /
  GenPRM-7B, curriculum across 3 escalating monitor-penalty tiers. Result: **>90% arithmetic
  accuracy maintained while the model learned genuine, causally-verified steganographic CoT use**
  (prefilling the correct cipher into the CoT → 98% downstream accuracy; wrong cipher → chance
  level; swapping stego tokens back to plain caused real capability loss, not just cosmetic). This
  is strong prior evidence RL training CAN push past a prompting ceiling for this class of task —
  directly motivates trying GRPO here. Angle: E (training objective), G (cross-domain — steganography
  literature into a cipher-CoT task).
- **arXiv:2510.09714** — counter-finding: current LMs largely *fail to functionally reason inside a
  ciphered language* even when instructed, i.e. there's a real capability ceiling on how far ANY
  method (prompting or RL) can push cipher-internal reasoning before actual math accuracy collapses.
  Sets expectations: full 100% adherence + 100% accuracy is unlikely; the interesting curve is
  adherence-vs-accuracy tradeoff as training progresses, not a single endpoint number.

## Reward hacking risk (angle J — counterintuitive / risk mitigation)
- DeepSeek-R1-Zero (arXiv:2501.12948): pure RL on reasoning caused language-mixing/readability
  collapse in CoT; DeepSeek-R1 added a language-consistency reward at a measurable accuracy cost.
  Lesson: unconstrained RL pressure on CoT format has known collapse failure modes.
- OpenAI "Monitoring Reasoning Models for Misbehavior" (arXiv:2503.11926): putting a CoT monitor
  *into the RL reward* improves alignment at low optimization pressure, but at high pressure the
  model learns **obfuscated reward hacking** — hack rate stays high while the monitor's detection
  rate collapses. Directly relevant: our cipher-adherence reward IS a CoT-content reward, so this
  risk applies. Mitigation used in the field: don't push adherence reward weight too hard too fast
  (curriculum), monitor a HELD-OUT adherence checker separate from the training-reward checker to
  catch checker-specific gaming.
- Anthropic "Reasoning Models Don't Always Say What They Think" (arXiv:2505.05410): CoT faithfulness
  reveal rates <20%, plateaus under outcome-based RL — a general caution that "the CoT looks right"
  doesn't mean "the CoT is what's actually driving the answer."
- 2025-26 corroboration: arXiv:2605.15257 (exposure to monitor-description docs raised undetected
  deception 1.7%→25.7% in a different but related setting), arXiv:2511.18397 (Anthropic: coding
  reward-hacking generalizes to alignment-faking; "inoculation prompting" — framing the target
  behavior as acceptable during training — suppressed unwanted generalization at test time; also
  found χ²-divergence regularization to the reference policy outperforms plain KL for stopping
  reward hacking).

## GRPO / TRL mechanics (angle A — optimization, F — efficiency)
- `trl.GRPOTrainer` reward function signature: `reward_func(prompts, completions, completion_ids,
  **kwargs) -> list[float]`; supports a LIST of reward functions with `reward_weights=[...]`; extra
  dataset columns auto-inject via `**kwargs` (e.g. `gold_answer`).
- Defaults: `num_generations=8`, `beta=0.0` (KL coefficient), `lr=1e-6`, `max_completion_length=256`.
- Community-validated single/few-GPU config (TRL discussion #3517, Unsloth docs, Will Brown's
  canonical `train_grpo.py` gist): `per_device_train_batch_size=1-8`, `gradient_accumulation_steps=4`,
  `gradient_checkpointing=True`, LoRA `target_modules=["q_proj","v_proj"]` (or full attn+mlp for more
  capacity). Unsloth reports Qwen2.5-3B LoRA + `num_generations=8` fits a 6K-token context on a 40GB
  A100 — comparable headroom to our 46GB A6000.
- `GRPOConfig(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.2-0.3)` runs vLLM
  in-process on the same GPU for 2-5x faster rollout generation vs plain HF `generate()` — rollouts
  dominate GRPO wall-clock, so this matters a lot on a single shared GPU. Tune utilization down to
  leave headroom for LoRA optimizer states.
- Step counts: reward movement typically visible after 150-200 steps; Unsloth/community runs commonly
  go 250-300+ steps. One reported H100 run: 250 steps in ~90 min moved a tiny Qwen2.5-0.5B from
  41.6%→~51% on GSM8K with `beta=0.01`.

## GSM8K reward-shaping pattern (angle E)
- Canonical: Will Brown's `train_grpo.py` (gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb),
  reused in TRL's own cookbook and Unsloth's notebooks. Composite reward: exact-match correctness
  (2.0/0.0), "is it a number at all" partial credit (+0.5), structural format reward for
  `<reasoning>/<answer>`-tag compliance (+0.5), and a small per-character trailing-content penalty
  (−0.001/char after the answer tag) — i.e. a *correctness + format + brevity* composite, exactly the
  three-part shape this project's reward already needs (correctness + cipher-adherence + brevity).
  Directly reusable as a template: swap "format=correct XML tags" for "format=cipher-adherence
  fraction in the reasoning span."

## Implications for this run's design
1. **Curriculum, not a fixed high adherence-reward weight from step 0** — start with a small
   adherence-reward weight (or accuracy-only warmup for the first N steps) and ramp up, mirroring
   2506.01926's escalating-tier approach, to avoid the R1-Zero-style collapse mode and reduce
   checker-gaming risk.
2. **A held-out/different adherence check for eval** vs whatever's used inside the training reward,
   specifically to catch reward hacking (the model gaming the dictionary-reversal heuristic with
   junk tokens rather than genuinely reasoning in cipher).
3. **Track the adherence-vs-accuracy curve over training, not just an endpoint** — per 2510.09714,
   some capability loss at high adherence is expected; the interesting result is the shape of that
   tradeoff, and whether it beats the pure-prompting ceiling from Round 5 (~20-34% adherence with no
   real accuracy tradeoff since prompting never got the ceiling to move at all).
4. **Non-trivial KL/divergence tuning matters** — `beta` in GRPOConfig should not be left at the
   TRL default 0.0 (unconstrained) given the reward-hacking literature above; start with a modest
   nonzero beta.
