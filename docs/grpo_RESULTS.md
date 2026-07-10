# RESULTS — GRPO training for cipher-reasoning and token-efficient reasoning

**⚠️ Known caveat on Round A below, added after Round B's post-mortem:** Round A used the same
`learning_rate=2e-6` that turned out to be a real bug in Round B (25-500x too low for LoRA GRPO —
see "Round B, corrected" further down for the full diagnostic). Round A's weight movement was 1.06%
over 260 steps, the same order of magnitude as Round B's confirmed-broken run. Round A's numbers
below should be read as a **lower bound** on what GRPO can do for cipher-reasoning, not a fully
resolved answer — it has not yet been rerun with a corrected LR.

## Round A: can GRPO push reasoning-trace cipher adherence past the prompting ceiling?

**Goal.** Round 5 of the parent session (`~/autoresearch-runs/llm-cipher-language/RESULTS.md`) used
GEPA (prompt optimization) to try to make a reasoning model (Kimi K2 Thinking) think — in its
internal chain-of-thought, not just its final answer — using a private letter-reversal cipher. That
hit a hard ceiling: the model's final visible answer reliably reached 94-100% cipher adherence, but
its internal reasoning trace stayed stuck at ~11-34% adherence across 40+ increasingly aggressive
prompt iterations. This round asks whether **RL training (GRPO)** — which puts reward pressure
directly on the sampled tokens of the reasoning span itself, unlike a system prompt — can break
that ceiling.

### Infra debugging saga
1. **verl not found.** The user expected `verl` (a GRPO-focused RL training framework) to be
   available on the target GPU host `eva02`. A full search (system Python + ~15 local venvs) found
   nothing. User chose to proceed with **TRL** (`trl==0.28.0` pre-installed, has `GRPOTrainer`)
   rather than install verl from scratch.
2. **CUDA version whack-a-mole.** Installing the rest of the stack (`transformers`, `peft`,
   `accelerate`, `datasets`, `bitsandbytes`, `vllm`) pulled in the latest `vllm`, which forced
   `torch` to a CUDA 13.0 build — incompatible with eva02's driver (575.64.03 / CUDA 12.9capable).
   Upgrading the shared box's driver was out of scope. Fix: dropped `vllm` entirely (using TRL's
   plain HF-`generate()` rollout backend instead of vLLM colocate mode — slower rollouts, zero
   infra risk), then manually pinned `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`,
   `torchaudio==2.6.0+cu124` to a mutually-compatible set (transformers 5.x eagerly imports
   `torchvision`/`torchaudio` even for a pure-text model, so all three had to match).
3. **Wrong model architecture.** The user asked for "Qwen3.5-4B." `Qwen/Qwen3.5-4B` turned out to
   be a **multimodal VLM** (`Qwen3_5ForConditionalGeneration`) with a novel hybrid linear-attention
   architecture requiring a `transformers` dev build — too much risk for LoRA+GRPO compatibility.
   Substituted `Qwen/Qwen3-4B-Instruct-2507` — same vendor/size class, pure-text
   `Qwen3ForCausalLM`, well-supported out of the box.
4. **Mid-training crash.** Phase 1 crashed at step ~16 with `OverflowError: cannot convert float
   infinity to integer` — a malformed generated number parsed to `float('inf')` inside the
   correctness reward's number-normalization code. Fixed with NaN/inf guards in `normalize_num`
   plus a `_safe()` wrapper around every reward function so one bad completion scores 0.0 instead
   of crashing the whole trainer (also cut `save_steps` 20→10 to limit any future lost progress).

### Curriculum design
Two-phase curriculum, following DEEPRESEARCH.md's synthesis of DeepSeek-R1-Zero (unconstrained RL
on CoT format caused language-mixing/readability collapse) and arXiv:2506.01926 (escalating-tier
reward curriculum let a model learn genuine steganographic CoT while keeping >90% task accuracy):
- **Phase 0** (40 steps): correctness + format reward only — warmup, establishes reliable
  `<reasoning>/<answer>` formatting and high baseline accuracy before any cipher pressure.
- **Phase 1** (260 steps): adds cipher-adherence + brevity reward on top, resumed from Phase 0's
  LoRA checkpoint.

### Training-time adherence trend (Phase 1, 260 steps, ~3h32m)
Adherence reward mean, tracked via periodic monitoring (rolling window averages):
`0.05 (baseline) → 0.21 → 0.27 → 0.28 → 0.23 → noisy, fluctuating ~0.02-0.72 per-batch through
steps 150-260`. Per-batch values were often elevated well above the ~0.05 starting point,
peaking as high as 0.72 in individual batches, but never settled into a stable monotonic climb —
more a noisy plateau than convergence. Correctness reward softened somewhat over the same period
(from near-2.0/2.0 in Phase 0 to averaging roughly 1.3-1.6/2.0 by late Phase 1) as cipher pressure
competed with task performance.

**Qualitative signal during training** was genuinely encouraging: at least one rollout (step ~84,
"call=164") showed the model spontaneously writing an explicit word→cipher **translation table**
as scratchpad before using it — `"one" -> "enoy"`, `"barnyard" -> "yradnab"`, `"owl" -> "wlo"` — a
real (if imperfect — several mappings were wrong) emergent strategy, not noise. Other high-scoring
samples looked like incoherent letter-salad exploiting the dictionary-based adherence checker's
sensitivity to low-`n_judged` flukes (a single lucky reversed word could spike the raw score) — a
real instance of the reward-hacking risk flagged in DEEPRESEARCH.md (OpenAI's "Monitoring Reasoning
Models for Misbehavior," obfuscated reward hacking under CoT-content reward pressure).

### The honest result: held-out eval at low temperature told a different story

Training-time rollouts used `temperature=0.9` for exploration, standard for GRPO. The **held-out
evaluation** (70 fresh GSM8K test examples, `temperature=0.3`, i.e. the model's actual settled
behavior) told a much more sobering story than the training curves suggested:

| Condition | Accuracy | Mean reasoning-cipher-adherence |
|---|---|---|
| (a) No cipher instruction (reference ceiling) | **95.71%** | 3.6% (noise floor) |
| (b) Cipher system prompt, base model, no training | 87.14% | 3.3% (≈ noise floor — untrained model barely attempts it at all) |
| (c) Cipher system prompt, GRPO-trained (Phase 1) | 84.29% | **6.4%** |

GRPO training roughly **doubled** adherence over the untrained baseline (3.3% → 6.4%) — a real,
non-zero effect — but landed nowhere near Round 5's ~20-34% prompting-ceiling on kimi-k2-thinking,
and cost a further ~3 accuracy points on top of the ~9-point cost prompting alone already imposed
(95.71% → 84.29% end to end).

**Reading the actual qualitative samples (15 full outputs, post-training, low temperature)**
resolved the genuine-vs-gamed question cleanly: **11 of 15 were plain, uncriphered English**
reasoning that happened to still be correct. Only 2-3 showed any cipher attempt at all, and those
followed a consistent pattern — the model would do a **partial, front-loaded gesture** toward the
cipher (an explicit word-translation-table preamble, or scrambling a handful of proper nouns/early
words) and then **revert to normal English** for the substantive arithmetic. This is neither
"genuine functional steganographic reasoning" (the model never did real math *inside* the cipher)
nor "pure incoherent reward-hacking noise" (the attempts that did occur used real, if error-prone,
letter-reversal) — it's best described as **learned partial/token-gesture compliance**: touch the
requirement just enough to bank some reward, then do the real cognitive work in the mode the model
is actually fluent in.

### Headline verdict — Round A

**GRPO training moved the needle, but did not remotely break the prompting-only ceiling from
Round 5, and the mid-training excitement (0.2-0.3 average adherence, a genuine translation-table
strategy) was substantially an artifact of high-temperature exploration sampling, not a stable
shift in the model's default policy.** At eval time, held-out adherence (6.4%) was only ~2x above
noise floor, well below Round 5's 20-34% figure on a different base model (Kimi K2 Thinking vs
Qwen3-4B-Instruct-2507 here — a real caveat on direct comparability, but the gap is large enough
that "GRPO solved what prompting couldn't" is not a fair characterization of this run). 260 steps
on a 4B model with this reward weighting was insufficient to overcome the model's strong "solve
math in plain English" prior. Plausible next levers (not attempted in this round): many more steps,
a bigger model, a stronger/curriculum-scheduled adherence weight (the correctness reward's max 2.0
likely dominated the adherence reward's max 1.5 throughout), or — as it turned out — **the wrong
objective entirely**, addressed in Round B below.

## Round B: pivot — token-efficient reasoning (not cipher-for-obfuscation)

Mid-writeup, the user reframed the actual goal: not obfuscation for its own sake, but **using a
private/compact encoding to genuinely save tokens** (and therefore inference cost/latency).

**Key technical correction that motivated the pivot:** the letter-reversal cipher used in Round A
is *counterproductive* to that goal. BPE tokenizers (including Qwen3's) already encode common
English words as single tokens (`"the"`, `"gave"`, `"half"` are each ~1 token); a letter-scrambled
form like `"eht"`, `"evag"`, `"lfaeh"` is out-of-vocabulary for the tokenizer's learned merges and
typically fragments into *more* tokens per character, not fewer. Round A's cipher, even if fully
adopted, would likely have made reasoning **longer** in token terms, not shorter — directly opposed
to what the user actually wants.

**Redesign (`grpo_train_efficient.py`):** dropped the letter-reversal requirement and the
dictionary-based adherence reward entirely. New reward = `format (0.5 max) + correctness (2.0 max,
gates everything below) + token_efficiency (1.5 max, computed via the model's real tokenizer on
the <reasoning> span, only paid out when the answer is correct so the model can't game brevity by
giving up)`. Token-efficiency reward: full credit at ≤40 reasoning tokens, linearly decaying to
zero at ≥150 tokens. System prompt asks for "radically concise... bare numbers, operators, and
minimal symbols only," dropping the cipher instruction but keeping the spirit of a terse,
non-natural-language notation as a side effect of genuine compression, not an explicit encoding
requirement.

**Smoke test (3 steps, resumed from the Phase 0 correctness+format checkpoint) was immediately
encouraging:** step 1 already produced `completions/mean_length=64.4` tokens (down from Round A's
Phase-0 baseline of ~150-300+ tokens for natural verbose CoT), `reward_correctness=2.0/2.0`
(unchanged, perfect), `reward_token_efficiency=1.334/1.5` (near max). Unlike the cipher objective,
"be concise" is a request the model can straightforwardly comply with using skills it already has,
rather than fighting an out-of-distribution letter-transformation task — this predicts a much
cleaner training run than Round A's.

Full 260-step run (resumed from the Phase 0 correctness+format checkpoint) completed in **~61
minutes** — 3.5x faster than Round A's 3h32m, simply because short completions generate faster.

### Training-time trend
Completion length stayed consistently low throughout — mostly 40-95 tokens per batch from early
steps onward (vs. Round A/Phase-0's natural ~150-300+ token baseline), with correctness holding in
the 1.0-2.0/2.0 range (averaging roughly 1.5-1.7) and the token-efficiency reward frequently near
its 1.5 max. Qualitative samples logged during training were consistently clean, genuine arithmetic
notation — e.g. `"8 → half = 4 → 4 + 10 = 14 spiders → 14 × 8 = 112"`,
`"8 + (8+4) + 2*(8+4) = 8 + 12 + 24 = 44"` — readable, mostly correct, with errors being ordinary
arithmetic mistakes rather than degenerate compression artifacts. A much healthier training signal
throughout than Round A's cipher confusion.

### Final held-out three-way comparison (70 examples, `temperature=0.3`)

**⚠️ SUPERSEDED — this run had a learning-rate bug.** See "Round B, corrected" below for the
authoritative result. Kept here for the audit trail, not as the real finding.

| Condition | Accuracy | Mean reasoning tokens |
|---|---|---|
| (a) Plain baseline, no conciseness instruction | **97.14%** | 197.2 |
| (b) Concise-notation prompt, base model, no training | 85.71% | 62.2 |
| (c) Concise-notation prompt, GRPO-trained (Round B, buggy LR) | 84.29% | 62.3 |

## Round B, corrected: the learning-rate bug

The user pushed back directly on the result above ("this sucks, debug the RL properly") — the
identical stats between (b) and (c) were suspicious enough to warrant real investigation rather
than reporting them as a genuine finding.

**Diagnostic: LoRA adapter weight L2-norm diff between checkpoints.** Comparing
`ckpt_phase0` (pre-Round-B) to `ckpt_phase_eff` (post-Round-B, 260 GRPO steps) directly at the
weight level: relative change was only **0.65%**. The same check on Round A's checkpoints
(`ckpt_phase0` → `ckpt_phase1`, also 260 steps) showed **1.06%**. In both cases, 260 real GRPO
steps barely moved the adapter at all — this explains both rounds' weak results far better than
any behavioral hypothesis about the task itself.

**Root cause:** `learning_rate=2e-6`, taken from GRPO literature written for **full fine-tuning**.
LoRA has orders of magnitude fewer trainable parameters and needs correspondingly larger per-step
updates — standard LoRA fine-tuning practice uses `1e-4`–`5e-4`, not `2e-6`. The original value was
25-500x too conservative.

**First fix attempt (LR=5e-5) overshot.** A relaunched Round B at 5e-5 showed real movement
immediately (1.8% weight change after just 5 steps — more than the entire old 260-step run) but
was unstable: `kl` repeatedly exceeded 0.5, peaking at 0.83, well past a healthy range. Killed
before completion rather than trusting an unstable run.

**Calibrated fix: LR=1.5e-5, beta=0.06** (beta raised from 0.04 for a firmer KL leash at the higher
LR). Validated via a 15-step smoke test: KL stayed in 0.0002-0.03, weight movement reached 0.9% in
just 15 steps (already close to the old buggy run's entire 260-step total). Full 260-step run
launched on this config, completed in **61 minutes**, KL stayed bounded throughout (0.003-0.15,
no sustained excursions past ~0.15) — a stable run. Final weight movement: **3.45%** relative
change vs. `ckpt_phase0` — 5x more than the buggy run, confirming real learning happened.

### Corrected final held-out three-way comparison (70 examples, `temperature=0.3`)

| Condition | Accuracy | Mean reasoning tokens |
|---|---|---|
| (a) Plain baseline, no conciseness instruction | 94.29% | 184.6 |
| (b) Concise-notation prompt, base model, no training | 84.29% | 63.2 |
| (c) Concise-notation prompt, **GRPO-trained, LR fixed** | **85.71%** | **58.3** |

With the bug fixed, condition (c) now **beats condition (b) on both axes simultaneously**: +1.4
accuracy points and 7.8% fewer tokens (58.3 vs 63.2) — a real, if modest, win-win rather than the
buggy run's statistical tie. A representative correct example (46 tokens):
```
50 * 1/2 = 25 → remaining
25 * 3/5 = 15 → given to Charlie
50 - 25 - 15 = 10
```
Qualitative samples throughout remained genuine, coherent terse arithmetic — no degeneration from
the higher LR.

### Headline verdict — Round B (corrected)

**Prompting alone captures most of the achievable gain (94.29%→184.6 tokens down to 84.29%→63.2
tokens from just asking for conciseness), but with the learning-rate bug fixed, GRPO training now
adds a real, measurable improvement on top: +1.4 accuracy points and 7.8% fewer tokens than
prompting alone, simultaneously.** The earlier "GRPO added nothing" conclusion was an artifact of
training barely happening at all (0.65% weight movement over 260 nominal steps) — not a genuine
finding about RL vs. prompting for this task. With real training (3.45% weight movement, stable
KL), the RL pass does buy something beyond the prompt, it's just a modest refinement rather than a
transformative gain, because the prompt-only baseline was already capturing most of the available
improvement (a model this well-instruction-tuned doesn't have much headroom left after "just ask it
to be concise").

**Revised, more accurate lesson from this pair of experiments:** RL training's *ceiling* is set by
how much the target behavior fights the model's priors (Round A's letter-cipher, which prompting
could barely touch, so RL's ceiling — even at Round A's still-unfixed low LR — was low too), but
whether RL reaches anywhere near that ceiling depends entirely on whether it's actually configured
to learn (LoRA needs a LoRA-appropriate LR, not one copied from a full-fine-tuning recipe). **Always
verify a "no effect" RL result by checking the adapter's actual weight movement before concluding
the objective itself doesn't respond to training** — a flat result is at least as likely to be a
broken training run as a genuine negative finding, and the two look identical from the eval numbers
alone.

**Round A almost certainly has the same bug and was never properly retrained.** Its weight movement
was 1.06% over 260 steps — same order of magnitude as Round B's broken run. Its reported ~6.4%
reasoning-adherence result (vs. Round 5's ~20-34% prompting ceiling on a different model) should be
read as a **lower bound**, not Round A's real capability — a properly-tuned rerun (same LR≈1.5e-5,
beta≈0.06 starting point, though cipher-adherence may need its own calibration pass given it's a
harder, more out-of-distribution objective than token-brevity) has not yet been done and is the
natural next step if the user wants the cipher question fully resolved rather than left on a
known-buggy number.

## Total compute for this GRPO round
Round A: ~4 hours (40-step warmup + 260-step main phase) on a single shared RTX A6000, eva02.
Round B: ~1 hour (260-step main phase, reused Round A's Phase 0 warmup checkpoint), same host.
No dollar cost for either round — this sub-project used the user's own GPU, unlike the API-based
Rounds 1-5 in the parent session (`~/autoresearch-runs/llm-cipher-language/`).

## Raw data
- `grpo_train.py` — Round A harness (letter-reversal cipher reward)
- `grpo_train_efficient.py` — Round B harness (token-efficiency reward)
- `final_eval.py` / `final_eval_result.json` / `qualitative_samples.json` — Round A's three-way
  held-out comparison and full qualitative samples
- `final_eval_eff.py` / `final_eval_eff_result.json` / `qualitative_samples_eff.json` — Round B's
  three-way held-out comparison and full qualitative samples
- `TASK.md`, `PREVIOUS_WORK.md`, `DEEPRESEARCH.md`, `COMPUTE.md`, `DATA.md`, `BUDGET.md` — the
  autoresearch-skill planning documents for this run
- On eva02: `~/grpo_cipher/ckpt_phase0` (warmup), `~/grpo_cipher/ckpt_phase1` (Round A result),
  `~/grpo_cipher/ckpt_phase_eff` (Round B, in progress), `phase0.log` / `phase1.log` /
  `phase_eff.log` (full training logs), `reward_log_phase*.jsonl` (periodic qualitative samples)

## Round A (scaled), out-of-domain evaluation: AIME 2026 + TerminalBench 2.0

Following the LR-bug fix (see above), Round A was rerun at scale: LoRA r=32, LR=1.5e-5, beta=0.06,
600 steps, **full 7473-example GSM8K training set** (no subsampling), batch=8×num_generations=16,
on the same shared RTX A6000 (eva02). This produced `ckpt_phase1_scaled`, ~9.5 hours of training.
Final training-time metrics (last logged step): correctness reward 1.875/2.0, cipher-adherence
reward 1.375/1.5, KL 0.13-0.16 throughout (stable, never approached the 0.3-0.4 instability
threshold used elsewhere in this project).

**Weight-movement sanity check** (same technique as the Round B LR-bug diagnosis: LoRA adapter
L2-norm diff via `safetensors.torch.load_file`, summed across all tensors, normalized by the
baseline's total norm): `ckpt_phase1_scaled` vs `ckpt_phase0_r32` (the r=32 warmup baseline) =
**3.54% relative movement** — consistent with real training having occurred (compare to the
<1% signature that flagged the original LR=2e-6 bug, and to Round B's corrected run at 3.45%).

The user then asked for two out-of-domain benchmarks to check whether the narrow GSM8K
cipher-reasoning fine-tune transfers anywhere: **AIME 2026** (harder, olympiad-level math) and
**TerminalBench 2.0** (general agentic terminal-use). Both were run base-vs-trained via a locally
hosted sglang server (Qwen3-4B-Instruct-2507 base model, `ckpt_phase1_scaled` LoRA adapter for the
trained condition) so the comparison uses the exact same serving stack for both conditions.

### AIME 2026 (30 held-out problems, MathArena dataset, temperature=0.3)

| Condition | Accuracy |
|---|---|
| Base (untrained, concise-cipher-style system prompt) | 36.67% (11/30) |
| Trained (`ckpt_phase1_scaled` LoRA) | 36.67% (11/30) |

**Identical accuracy, and not just coincidentally close — both conditions got exactly 11/30
right.** GRPO fine-tuning narrowly on GSM8K-level (grade-school) cipher-reasoning produced no
measurable change, positive or negative, on held-out olympiad-level math. This is plausible on its
face: AIME-level problems are far outside the training distribution's difficulty range, and a
LoRA adapter tuned on an easier task has no particular reason to help or hurt a much harder one.

### TerminalBench 2.0 (8-task representative subset, harbor's terminus-2 reference agent)

Tasks: `fix-git`, `regex-log`, `sqlite-db-truncate`, `password-recovery`, `large-scale-text-editing`,
`nginx-request-logging`, `openssl-selfsigned-cert`, `count-dataset-tokens` (chosen as a disk- and
time-bounded subset of the full 89-task Terminal-Bench 2.0 registry — see infra note below on why
disk was a real constraint on this shared host).

| Condition | Mean reward | Breakdown |
|---|---|---|
| Base | 0.0 / 8 | 6 genuine task failures + 2 `AgentTimeoutError` |
| Trained (`ckpt_phase1_scaled` LoRA) | 0.0 / 8 | 6 genuine task failures + 2 `AgentTimeoutError` |

Both conditions failed every task in the subset. Real token usage was confirmed for both runs
(millions of input tokens, tens of thousands of output tokens across the 8 trials each) — these
are genuine multi-turn agent attempts, not short-circuited errors. This should **not** be read as
"the cipher training broke agentic ability" — the base (untrained) condition failed identically,
so there is no differential effect either way. It more likely reflects that terminus-2 (designed
around larger, more capable models) is a hard harness for a 4B model in general, independent of
this fine-tune.

### Infra debugging notes (for reproducibility)

Standing up local sglang serving + harbor evaluation on eva02 required three fixes, none related
to the trained model itself:
1. **flashinfer JIT compile failure**: sglang's default attention backend JIT-compiles CUDA
   kernels via nvcc at first request; this failed with `fatal error: math.h: No such file or
   directory` from inside `#include_next` despite the file existing — a known nvcc/gcc header
   include-order quirk on this host's gcc-13/CUDA-12.0 combination. Fixed by passing
   `--attention-backend triton` to `sglang.launch_server`, which avoids the flashinfer CUDA JIT
   path entirely.
2. **harbor/litellm missing credentials**: TerminalBench2 trials all failed with
   `InternalServerError` → `Missing credentials... OPENAI_API_KEY`. Harbor's `--ae` flag only
   injects env vars into the sandboxed *agent's* execution environment, not into harbor's own
   host-side litellm client that actually places the API call. Fixed by setting
   `OPENAI_API_KEY=dummy` as a real process env var on the `harbor run` invocation itself.
3. **sglang LoRA + radix-cache incompatibility**: the trained condition's server crashed on
   startup with `AssertionError: compatibility of lora and radix attention is in progress`. Fixed
   by adding `--disable-radix-cache` alongside `--lora-paths`/`--enable-lora`.

### Verdict

Combined with the corrected in-domain GSM8K result above (Round B: GRPO training measurably beats
prompting-only, +1.4pp accuracy / -7.8% tokens), this scaled Round A run shows the cipher-reasoning
GRPO training produces **real in-domain movement with no detectable transfer — positive or
negative — to harder math or general agentic tasks**. For a LoRA adapter trained narrowly on
grade-school-level cipher-reasoning, this is an honest, plausible outcome rather than a failure
that demands further tuning: the training objective simply never touched anything resembling
AIME-level math or shell/tool-use, so no transfer (in either direction) is exactly what a
faithful measurement should show.
