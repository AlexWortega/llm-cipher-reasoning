# TASK

Direct follow-up to `~/autoresearch-runs/llm-cipher-language/RESULTS.md` (Round 5, "GEPA cipher
reasoning"). That experiment used prompt optimization (GEPA) to try to make Kimi K2 Thinking reason
*internally* using a private letter-reversal cipher while solving GSM8K math. Result: the model's
**final visible answer** reliably hit 94-100% cipher adherence, but its **internal `reasoning`
field** stayed stuck at ~20-34% adherence across 40+ increasingly aggressive prompt iterations —
looked like a genuine prompting/steerability ceiling, not a wording problem.

**This round's question:** can RL training (GRPO) push past that ceiling? Unlike a system prompt,
GRPO puts reward pressure directly on the sampled tokens of the reasoning span itself, via relative
advantage within a group of rollouts per prompt. Hypothesis: reasoning-trace cipher adherence can be
pushed well above the ~20-34% prompting ceiling through a few hundred GRPO steps, and the interesting
question is what happens to accuracy as adherence rises (does the model learn to actually solve math
*in* the cipher, or does correctness collapse as reasoning gets forced into an unnatural format).

**Run mode:** interactive, but the user is not actively watching every step — standing autonomy
authorization applies (established over this whole session) once compute+data are confirmed.

**Unknowns going in:**
- Exact correct HF model ID for "Qwen3.5-4B" (verify before downloading)
- Real achievable GRPO throughput/step time on a single 46GB A6000 with LoRA
- Whether reward hacking emerges (e.g. model games the adherence checker with junk-but-reversible
  tokens instead of real reasoning)
- Whether verl was meant to be available (user thought so; not found on eva02 after full search
  across ~15 local venvs + system python) — **user chose to proceed with TRL** (already installed,
  `trl==0.28.0`, has `GRPOTrainer`), not to install verl fresh.

**Admits many hypotheses worth sweeping:** yes — reward-shaping weights, cipher rule choice
(word-reversal vs alternatives), LoRA rank/target-modules, KL penalty strength, group size, and
whether to warm-start from the GEPA-optimized system prompt vs a neutral one are all independent
levers worth comparing on a fixed compute budget.
