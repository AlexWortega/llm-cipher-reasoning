# DEEPRESEARCH — Round E: non-verbal, token-shorter reasoning

Three parallel research agents surveyed: (1) continuous/latent thought-vector methods, (2)
pause/filler tokens + learned discrete codebooks, (3) engineering feasibility of the top repos
against our TRL GRPOTrainer + LoRA + single-GPU pipeline.

## A. Continuous/latent "thought vector" methods (angle C — architecture)

| Method | arXiv | Mechanism | GSM8K result | Training | Feasibility for us |
|---|---|---|---|---|---|
| **Coconut** (Meta/UCSD) | 2412.06769 | last hidden state fed back as next input embedding | 34.1% vs 42.9% CoT (**loses on math**) | SFT-only, multi-stage curriculum | GPT-2-scale only, no LoRA/RL shown |
| **CODI** | 2502.21074 | self-distillation, teacher+student heads aligned | 43.7% ≈ 44.1% CoT, 3.1x compression | SFT/self-distillation, LoRA(r128) | untested at 4B/instruct scale |
| **SoftCoT** | 2502.12134 | frozen target LLM + small trainable projection from assistant model | native Qwen support | SFT of projector only | **NTU non-commercial license — blocker**; SFT-only, no RL |
| **CoLaR** | 2505.16552 | compressed-embedding latent head + **GRPO RL stage** | 53.3% shorter chains, only 4.8% acc loss vs CoT | SFT then GRPO (custom, not TRL) | PyTorch Lightning + custom GRPO reimpl (`tiny-grpo`-derived), rollouts over continuous latents — **incompatible with TRL's discrete completion/reward pattern without a rollout-internals rewrite**. Only Llama-3.2-1B validated. Verdict: days of work, not hours. |

**RL-compatibility warning** (arXiv:2512.11816): naive GRPO on raw latent reasoning **fails** —
72.6% text-space GRPO vs only 21.8% latent-SFT+GRPO on Qwen2.5-1.5B/GSM8K, because outcome-only
reward gives no gradient signal for what happens inside latent steps. Newer patches (Latent-GRPO,
SofT-GRPO, both arXiv 2026) exist but are too new/unverified to build on.

**Verdict: deprioritized for Round E.** The one RL-proven method (CoLaR) needs a custom rollout
loop days of engineering; the rest are SFT-only and unproven past GPT-2/1B scale. Kept as a Next
Lever (rung-3 candidate) if the primary hypothesis below stalls.

## B. Pause/filler tokens (angle A/K) — ruled out

- **"Let's Think Dot by Dot"** (2404.15758): filler tokens replace CoT 1:1 (same length, not
  shorter); authors state RL/outcome-only discovery does not work, needs dense supervision.
- **"Think before you speak"** (2310.02226): pause tokens are *appended*, lengthening the
  sequence; gains required pretraining-time insertion, finetune-only injection gave no consistent
  benefit; GSM8K gain only +1% even in the best case.

**Verdict: ruled out entirely** — wrong direction (doesn't shorten) and not RL/finetune-transferable
per the original authors' own findings.

## C. Learned discrete codebooks / new vocabulary tokens (angle C/I) — primary lever

- **Token Assorted** (Meta/Berkeley, 2502.03275): VQ-VAE compresses early reasoning steps into new
  latent vocab tokens (genuinely new, not reused BPE). SFT with randomized latent/text-mixing
  curriculum, GSM8K/MATH gains with shorter traces. No RL. Needs an SFT warm-start before any GRPO
  polish could apply — cold-start RL over freshly-initialized token embeddings has no dense signal.
- **Shorthand for Thought** (Writer Inc., 2604.26355): mines frequent cross-word BPE merges
  ("supertokens") **from the model's own sampled reasoning traces** (not hand-designed) into new
  vocab entries; brief SFT teaches usage. 8.1% trace shortening, no accuracy loss, across 3 model
  families / 5 math benchmarks. **No public repo** (too recent), but the method is simple enough to
  reimplement from scratch: standard BPE-merge-style frequency counting over existing traces
  (~50-100 LOC), no external framework needed.

**Why this is different from our 4 failed prior attempts** (letter-reversal cipher, bit/n-ary
Huffman, stopword deletion, multi-token-word-avoidance reward): all four reused/rearranged the
**existing, already-near-optimal BPE vocabulary**. Token Assorted and Shorthand-for-Thought instead
**add brand-new vocabulary entries mined from the model's own high-frequency multi-token spans**,
with embeddings initialized from the constituent tokens (smooth warm start) — this is a genuinely
different lever, not a variation on the same one.

## Engineering feasibility verdict (3rd research agent)

| Candidate | Framework | RL reusable via TRL? | License | Verdict |
|---|---|---|---|---|
| CoLaR | PyTorch Lightning + custom GRPO | No — continuous-latent rollouts, would need TRL internals rewrite | Apache-2.0 | days of adaptation |
| SoftCoT | fastNLP, SFT-only | No RL in repo at all | **NTU non-commercial — blocker** | ~1 day for projector only, license risk |
| Shorthand for Thought | no repo (too new) | **Yes** — stays in discrete token space, drops straight into our existing `GRPOTrainer` + reward-function pattern | — (reimplement from paper description) | **<1 day, no framework rewrite** |

## Recommendation

Build the **Shorthand-for-Thought-style supertoken approach** as Round E's primary experiment:
1. Mine top-K frequent multi-BPE-token n-grams from the model's own compact reasoning (Round
   B/D's qualitative samples + a fresh larger sample from `ckpt_phase_eff3`).
2. Add each as one new special vocabulary token; initialize its embedding as the mean of its
   constituent tokens' embeddings (smooth warm start, avoids cold-start RL problem flagged above).
3. Brief SFT warm-start: re-tokenize existing correct reasoning traces, replacing mined n-gram
   occurrences with the new token, teach the model to reproduce this form.
4. GRPO polish: reuse Round B's proven `reward_correctness` + `reward_token_efficiency`, unchanged.

This satisfies all three of the user's constraints simultaneously by construction: shorter (each
supertoken replaces 2-4 BPE tokens with 1), non-human (the new tokens are opaque IDs with no
English surface form — they don't decode back to readable words), and accuracy-preserving by design
(smooth embedding init + SFT warm-start avoids the cold-start collapse that would hit a
naive-random-token RL-only approach).

Continuous-latent methods (CoLaR in particular) remain a valid **Next Lever** if this stalls, but
require a custom rollout/reward loop outside TRL — correctly rung-3 (structural reframe) territory,
not a first move.
