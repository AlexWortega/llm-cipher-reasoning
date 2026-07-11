# TASK — Round E

User request (verbatim intent): "довести до ума non-language reasoning" — get the model to write
reasoning that (a) shortens generation length, (b) preserves answer quality/accuracy, and (c) is
**not human words** (not English, not natural-language-shaped notation).

**Why this is a new round, not a repeat of B/D:** four prior rounds already tested the "hand-design
a compact code, then train/prompt the model into it" family and it kept losing to plain English
under BPE tokenization:
- Round A: letter-reversal cipher — adherence trainable but never validated for token savings
  (BPE reverses common words into MORE tokens, not fewer)
- Huffman coding (bit-string and n-ary CJK-symbol variants, scratchpad-only, never trained) — both
  made token count WORSE (2.71x and 1.32x) than plain text
- Round B: direct token-count reward (no notation constraint) — the one clear win, +1.4pp accuracy
  / -7.8% tokens vs prompting alone, but the model's own discovered "compact" style is still English
  words/digits/operators, not non-human symbols
- Round D: reward_avoid_multitoken_words (push away from BPE-multi-token English words) — null
  in-domain result, and a real negative transfer on AIME-2026 (23.33% vs 36.67% base)

**The pattern across all four:** every attempt to hand-design or reward-shape a *discrete-vocabulary*
substitute for English keeps failing, because a modern BPE tokenizer already assigns ~1 token to
almost every common English word — there is no discrete token-level encoding of the *same
information content* that beats it. Non-human discrete symbols (CJK glyphs, cipher text, deleted
stopwords) either cost the same or more per token than English, so "fewer tokens" and "non-human
symbols" have been fighting each other, not reinforcing each other, in every experiment so far.

**Round E's actual question:** is there a way to get "shorter generation + preserved quality +
non-human-word reasoning" where the three goals reinforce rather than fight? The working hypothesis
going in (to be checked against literature before committing): the answer is not a *better discrete
code* (four rounds of evidence says no headroom there) but a *structural* change — reasoning that
never gets tokenized into human-readable text at all (e.g. continuous/latent "thought vectors" fed
back as input embeddings instead of decoded to text), which mechanically satisfies all three
constraints at once: it's non-human by construction (not text), it can be shorter than any
text-token count (one continuous step can carry more than one token's worth of information), and
several published methods (surveyed below before committing compute) report preserved or improved
accuracy over word-token CoT on comparable math/reasoning benchmarks.

**Run mode:** headless-autonomous — continuing the same standing project (`~/grpo_cipher/` on
eva02) with the same standing authorization established across Rounds A-D this session. Compute
(eva02, single shared RTX A6000) and data (GSM8K, already local at `~/grpo_cipher/`) are already
settled from prior rounds — not re-asking the user for these.

**Unknowns going in:**
- Whether TRL's GRPOTrainer (already in use, `trl==0.28.0`) supports training with continuous/latent
  reasoning steps out of the box, or whether this needs a custom training loop bypassing GRPOTrainer's
  standard text-sampling assumption
- Real compute cost of a latent-reasoning approach on a single 46GB A6000 with LoRA vs the ~1-2hr
  budget prior GRPO rounds used
- Whether a cheaper structural lever exists (e.g. explicit non-semantic filler/pause tokens, shown in
  the literature to help reasoning while trivially satisfying "not human words") that's worth trying
  before committing to the harder latent-reasoning implementation

**Admits many hypotheses worth sweeping:** yes — this round explicitly needs literature survey before
locking the plan, per the skill's research-before-code rule; multiple structural angles (latent
thought vectors, filler/pause tokens, learned discrete codebook via VQ, curriculum distillation from
verbose to compressed reasoning) are plausible and should be compared rather than assumed.
