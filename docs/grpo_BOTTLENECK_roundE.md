# BOTTLENECK — Round E

## Bottleneck: B-ARCH (representation capacity), not B-OPT/B-DATA

Evidence: four independent prior attempts to shrink reasoning tokens by rearranging the
**existing** BPE vocabulary all failed or gave null/negative results — letter-reversal cipher
(never beat plain text on tokens), bit-Huffman (2.71x worse), n-ary CJK-symbol Huffman (1.32x
worse), stopword deletion (SFT abandoned, never trained), multi-token-word-avoidance reward
(Round D: null in-domain, -13.3pp on AIME transfer). All four had healthy training dynamics (real
KL movement, real reward signal) when actually trained — ruling out B-OPT (these weren't broken
runs). The common failure mode is structural: a modern BPE tokenizer already assigns ~1 token to
almost every common English word, so no re-encoding *within the same vocabulary* can beat it —
this is a capacity ceiling of the representation, not a tuning problem.

Angle constraint: architecture/vocabulary-extension experiments (angle C) get priority; forbidden
until this is retested: another pure reward-shaping experiment over the *existing* vocabulary
(angle A) with no structural change — that lever is exhausted per 4 independent negative/null
results and would just be experiment #5 in the same family.

One angle-A experiment is still permitted as a cheap confirmatory control (not the primary bet):
tightening `TARGET_TOKENS` further on Round B's existing reward stack, to rule out "the ceiling
was just under-pressured" before committing full effort to the vocabulary-extension approach.

## Angle priority in PLAN.md
- C (architecture / vocab extension) ≥ 1 experiment, this is the primary bet
- A (reward-shaping on existing vocab) ≤ 1 experiment, confirmatory control only
- G (cross-domain transplant — continuous latent methods) recorded as Next Lever, not attempted
  this round (engineering cost too high for the bounded budget per DEEPRESEARCH findings)
