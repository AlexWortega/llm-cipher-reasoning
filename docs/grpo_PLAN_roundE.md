# PLAN — Round E

<!-- BOTTLENECK: B-ARCH — angle-A cap: 1 (confirmatory only), angle-C floor: 1 (primary bet) -->

Compute/data unchanged from Rounds A-D: eva02 (single shared RTX A6000), GSM8K
(`~/grpo_cipher/gsm8k_train.jsonl` / `gsm8k_test.jsonl`), same standing authorization. Not
re-asking the user — this is a continuation of the same standing project.

Single-pass matrix (GPU is a single shared resource — real parallelism happens in
research/mining/code-writing, not concurrent GPU jobs; experiments run sequentially on eva02, same
pattern as Rounds A-D).

## Experiment 0 (control, angle A — cheap, run first while building Exp 1's mining pipeline)

**Change:** lower `reward_token_efficiency`'s `TARGET_TOKENS` from 40 to 20 on Round B's exact
existing setup (`ckpt_phase_eff3`'s reward stack, no new mechanism). 150 steps (half of Round B's
budget — this is a confirmatory probe, not a full round).

**Mechanism:** if the existing-vocabulary ceiling (Round D's finding) is real, tightening the
token target further should NOT produce meaningfully fewer tokens without an accuracy collapse —
confirms there's no more headroom to extract by reward-pressure alone before committing full
engineering effort to Experiment 1.

**Expected delta:** mean tokens roughly flat or accuracy drops sharply (either result confirms the
ceiling); a genuine token drop with accuracy held would falsify the B-ARCH bottleneck call.

**Falsification:** if mean tokens drop >15% with accuracy within 2pp of Round B's 85.71%, the
ceiling claim is wrong and Experiment 1's extra engineering effort was unnecessary.

## Experiment 1 (primary bet, angle C — vocabulary extension)

**Change:** mine top-K frequent multi-BPE-token n-grams from the model's own compact reasoning
traces (Round B/D qualitative samples + a fresh larger sample generated from `ckpt_phase_eff3` on
GSM8K train), add each as a new opaque special vocabulary token (no human-readable surface form on
decode), initialize new embeddings as the mean of constituent original-token embeddings, resize
the model's embedding matrix. Brief SFT warm-start (re-tokenize existing correct reasoning traces,
substituting mined n-gram spans with their new token) to teach fluent usage. Then GRPO polish
reusing Round B's exact `reward_correctness` + `reward_token_efficiency` (proven, unchanged) on
top of the SFT-seeded checkpoint.

**Mechanism:** unlike the 4 prior discrete-substitution attempts (which reused the existing
near-optimal BPE vocabulary), these are genuinely NEW tokens carrying more information per token
than any existing single BPE token can — each collapses 2-4 old tokens into 1. Smooth embedding
initialization + SFT warm-start avoids the cold-start problem that made naive RL-only symbol
invention fail elsewhere in the literature (arXiv:2512.11816).

**Expected delta:** mean reasoning tokens meaningfully below Round B's 58.3 (aim: -15 to -25%
further, i.e. ~44-50 tokens), held-out accuracy within ~2pp of Round B's 85.71%, and non-trivial
usage rate of the new supertokens in the final trained model's outputs (not reverted to 100%
original vocabulary).

**Falsification:** if post-SFT accuracy collapses before GRPO even starts, the embedding
initialization / mining choice was bad. If GRPO usage of new tokens converges to ~0%, the reward
stack doesn't actually incentivize them (would need a dedicated usage-bonus reward — noted as a
fallback variant, Experiment 1b).

**Fallback variant (1b, only if 1's supertoken usage collapses to near-zero):** add a small
`reward_symbol_usage` term (bonus proportional to the count of new-vocab tokens used in a correct
completion) to counteract any implicit bias toward the pretrained-familiar original vocabulary.

## Verification

Both experiments get the same 3-way held-out eval pattern used in Rounds B-D (70 examples,
`temperature=0.3`): plain baseline / concise-prompt-only / trained, reporting accuracy + mean
reasoning tokens (+ for Experiment 1: new-supertoken usage rate per completion, and a decode of a
handful of qualitative samples to confirm the reasoning is genuinely non-human-readable in the
supertoken spans, not just cosmetically renamed English).

## Next levers (if both experiments stall)

- CoLaR-style continuous-latent + GRPO (rung 3, structural reframe) — needs a custom rollout loop
  outside TRL, days not hours; revisit only if Experiment 1 and its 1b fallback both fail.
- Token Assorted's VQ-VAE approach as an alternative mining/quantization method for Experiment 1's
  token-selection step, if frequency-based n-gram mining proves too coarse.
