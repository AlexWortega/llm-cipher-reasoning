## Current baseline (from GEPA prompt-optimization round, same session)
- No-cipher reference ceiling: kimi-k2-thinking, plain GSM8K, 96.67% accuracy (58/60 held-out),
  $0.10 (source: `~/autoresearch-runs/llm-cipher-language/gepa_run/baseline_no_cipher_result.json`)
- Naive cipher prompt (unoptimized "reverse every word" instruction): 51.67% accuracy (31/60)
- GEPA-optimized cipher prompt (candidate 22 of 23 tried): 93.33% accuracy (56/60), avg score 0.572
  (source: `gepa_result.json`, `final_eval_result.json`)
- **Reasoning-field cipher adherence stuck at ~11-34% across all 23 GEPA candidates and 40+
  iterations**, while content-field (final visible answer) adherence reliably hit 94-100%. This is
  the ceiling this round is trying to break through with RL instead of prompting.

## What was tried and kept
- Letter-reversal cipher rule: reverse letters within each word, numbers/punctuation/spaces
  untouched. Validated as learnable-in-principle (content field proves the model CAN apply it
  correctly when it tries) — the barrier is specifically the reasoning trace, not the transform
  itself being too hard.
- Automatic adherence scorer using a dictionary heuristic (`cipher_utils.py`:
  `cipher_adherence_score()` — checks whether a word's reverse is a real dictionary word and the
  word itself isn't). Reusable as-is for the RL reward function.
- Few-shot worked examples embedded directly in the prompt (GEPA's best mutation) — meaningfully
  improved accuracy under the cipher constraint (51.67% → 93.33%) even though it didn't move
  reasoning-field adherence. Worth reusing as a warm-start system prompt / SFT-style seed for RL
  rollouts, since it establishes the *format* even if not the *internal* cipher use.

## What was tried and dropped
- Opus 4.6 as the GEPA reflection LM: reliably refused to propose cipher-instruction mutations
  (`content_filter` refusal), crashing the optimizer. Switched to Grok 4.5 (zero refusals all
  session). Not directly relevant to RL training, but the refusal pattern is worth watching for if
  any Anthropic model is used anywhere in this pipeline (e.g. as a reward-model judge).
- `VAL_N=70` / `MAX_METRIC_CALLS=8000` GEPA config projected a 28-hour ETA on kimi-k2-thinking API
  calls — throughput problem, not directly applicable to local GPU training, but the general lesson
  ("verify real per-step wall-clock before committing to a large step budget") carries over.

## Open hypotheses / TODOs
- Does the reasoning-vs-content adherence gap reflect something architectural in kimi-k2-thinking's
  RL-trained "thinking" mechanism specifically (i.e. might not generalize to a from-scratch GRPO run
  on a *different*, smaller model like Qwen3.5-4B that has no such pre-existing thinking/no-thinking
  split)? This round's base model doesn't have the same instruct-tuned "hidden good-faith thinking"
  prior kimi-k2-thinking does, so the ceiling might not even apply the same way. Worth noting
  explicitly as a reason results here may not be directly comparable to Round 5's number.
- Reward hacking risk: the dictionary-based adherence checker could be gamed (e.g. outputting short
  words, or exploiting palindrome/ambiguous-word edge cases the checker skips). Needs monitoring
  during training, not just a final adherence number.

## Known fragile areas
- No local GPU training infrastructure existed before this round — first GRPO run on eva02 in this
  project. Environment needs CUDA-enabled torch (currently CPU-only on eva02) and verified model/
  dataset availability before any training starts.
- Disk on eva02: 109GB free of 3.6TB (97% full) — model downloads and checkpoints must be managed
  carefully; no room for multiple full checkpoints of even a 4B model plus optimizer states without
  active cleanup.
