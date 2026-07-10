# llm-cipher-reasoning

Can LLMs invent a private/compact language for reasoning, and can that be trained in with RL
instead of just prompted? Two linked lines of research:

1. **Cipher invention / cross-model communication** (`gepa/`, rounds documented in `docs/RESULTS.md`):
   cold-decoding tests, negotiated cipher collusion between model pairs, a cipher-hardening arms
   race, and GEPA-based prompt optimization to get a model (Kimi K2 Thinking) to reason internally
   in a letter-reversal cipher via prompting alone.
2. **GRPO RL training for token efficiency** (`src/`, documented in `docs/grpo_RESULTS.md`):
   training Qwen3-4B-Instruct-2507 with GRPO (TRL) so the model *learns* — rather than is merely
   prompted — to reason in a compact form. Round A trains toward the letter-reversal cipher; Round B
   pivots to directly rewarding fewer reasoning tokens (the cipher itself turned out not to save
   tokens under BPE tokenization — see `docs/grpo_RESULTS.md`).

## Layout

```
src/            training + eval scripts (rounds 1-5 cipher experiments, GRPO training/eval)
gepa/           GEPA prompt-optimization run (cipher_utils, optimizer script, candidates, logs)
data/           gsm8k_train.jsonl, gsm8k_test.jsonl, dict_words.txt (cipher-adherence dictionary)
results/        all *_result.json / qualitative_samples*.json outputs
logs/           raw run logs (arms race rounds; GRPO reward_log_phase*.jsonl lands here too)
docs/           TASK/RESULTS/DEEPRESEARCH/PREVIOUS_WORK/BUDGET/COMPUTE/DATA writeups
checkpoints/    (gitignored, empty locally) — LoRA adapters are large and live on the remote
                GPU host (eva02) at ~/grpo_cipher/ckpt_*; not synced into this repo
.env            (gitignored) OPENROUTER_API_KEY, read by src/ and gepa/ scripts via ../.env
```

## Remote training host

GRPO training runs on `eva02` (via `ssh eva02`, ProxyJump `eva01`), in `~/grpo_cipher/` there —
a separate deployed copy of `src/grpo_train*.py`, with its own `.venv`, `hf_cache/`, and
checkpoint directories (`ckpt_phase0_r32`, `ckpt_phase1_scaled`, `ckpt_phase_eff3`, ...). The
copies in this repo's `src/` are the source-of-truth versions with paths adapted for local repo
layout (`ROOT_DIR/data`, `ROOT_DIR/checkpoints`, `ROOT_DIR/logs`) — redeploy by scp'ing to eva02
if you want the remote host to pick up a script change; the currently-running remote job is not
affected by edits here.

See `docs/grpo_RESULTS.md` for the current status of the GRPO runs, including a documented
learning-rate bug (LoRA needs ~1e-4-scale LR, not the 2e-6 copied from full-fine-tuning
literature) and its fix.
