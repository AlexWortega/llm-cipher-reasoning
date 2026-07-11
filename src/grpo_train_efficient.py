import json
import os
import re
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
PHASE = os.environ.get("PHASE", "eff")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "260"))
RESUME_FROM = os.environ.get("RESUME_FROM", "")  # LoRA adapter dir to resume from, or ""
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_DIR, "checkpoints", f"ckpt_phase{PHASE}"))
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "40"))  # full credit near/under this
MAX_CREDIT_TOKENS = int(os.environ.get("MAX_CREDIT_TOKENS", "150"))  # zero credit at/above this

_COUNT_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)

def normalize_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        if not (f == f) or f in (float("inf"), float("-inf")):
            return s
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s

def extract_answer_number(text):
    m = re.search(r"<answer>\s*(-?[\d,]+(?:\.\d+)?)\s*</answer>", text, re.IGNORECASE)
    if m:
        return normalize_num(m.group(1))
    nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", text)
    return normalize_num(nums[-1]) if nums else None

def extract_reasoning_span(text):
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""

WORD_RE = re.compile(r"[A-Za-z']+")
_MULTITOKEN_CACHE = {}

def word_token_cost(word):
    """Extra BPE tokens a word costs beyond 1 (0 for single-token words). Cached per lowercased
    word since the same common words recur constantly across a batch."""
    lw = word.lower()
    cached = _MULTITOKEN_CACHE.get(lw)
    if cached is not None:
        return cached
    n = len(_COUNT_TOKENIZER.encode(" " + lw, add_special_tokens=False))
    cost = max(0, n - 1)
    _MULTITOKEN_CACHE[lw] = cost
    return cost

def has_valid_format(text):
    return bool(re.search(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL))

def gsm8k_gold(answer_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return normalize_num(m.group(1)) if m else None

def load_gsm8k_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gold = gsm8k_gold(d["answer"])
            if gold is not None:
                rows.append({"question": d["question"], "gold": gold})
            if limit and len(rows) >= limit:
                break
    return rows

# NOTE: dropped the letter-reversal cipher requirement -- it doesn't actually save tokens (BPE
# tokenizers already encode common English words as single tokens; scrambled letters usually
# tokenize WORSE, into more single-character pieces). Reframed goal: directly reward fewer tokens
# in the reasoning span while keeping the answer correct. A terse symbolic notation is a natural
# byproduct and reads as non-obvious shorthand to a human, but the reward now targets the actual
# stated goal -- token savings -- not obfuscation for its own sake.
SYSTEM_PROMPT = """You are solving grade-school math word problems.

Solve the problem, then respond in exactly this structure:
<reasoning>
your reasoning here
</reasoning>
<answer>
the final numeric answer, digits only
</answer>

CRITICAL RULE for the <reasoning> section: be radically concise. Do not write in full sentences \
or natural-language explanations. Use the shortest possible notation -- bare numbers, operators, \
and minimal symbols only -- to get from the problem to the answer. Every unnecessary token costs \
you. Aim for the fewest possible tokens while still reaching the correct answer."""

def build_dataset(rows):
    data = []
    for r in rows:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
        ]
        data.append({"prompt": prompt, "gold": r["gold"]})
    return Dataset.from_list(data)

LOG_PATH = os.path.join(ROOT_DIR, "logs", f"reward_log_phase{PHASE}.jsonl")
_log_f = open(LOG_PATH, "a")

def _safe(fn, default=0.0):
    def wrapped(text):
        try:
            return fn(text)
        except Exception as e:
            print(f"[reward-safety] caught {type(e).__name__}: {e} -- scoring 0.0 for this completion")
            return default
    return wrapped

def reward_format(prompts, completions, **kwargs):
    scores = []
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        scores.append(_safe(lambda t: 0.5 if has_valid_format(t) else 0.0)(text))
    return scores

def reward_correctness(prompts, completions, gold, **kwargs):
    scores = []
    for c, g in zip(completions, gold):
        text = c if isinstance(c, str) else c[0]["content"]
        def _score(t, g=g):
            pred = extract_answer_number(t)
            return 2.0 if (pred is not None and pred == g) else 0.0
        scores.append(_safe(_score)(text))
    return scores

def reward_token_efficiency(prompts, completions, gold, **kwargs):
    """Reward fewer tokens in the reasoning span. Only pays out on CORRECT answers, so the model
    can't just game brevity by giving up early with a wrong/empty answer."""
    scores = []
    for c, g in zip(completions, gold):
        text = c if isinstance(c, str) else c[0]["content"]
        def _score(t, g=g):
            pred = extract_answer_number(t)
            if pred is None or pred != g:
                return 0.0
            reasoning = extract_reasoning_span(t)
            n_tokens = len(_COUNT_TOKENIZER.encode(reasoning, add_special_tokens=False))
            if n_tokens <= TARGET_TOKENS:
                return 1.5
            if n_tokens >= MAX_CREDIT_TOKENS:
                return 0.0
            frac = 1.0 - (n_tokens - TARGET_TOKENS) / (MAX_CREDIT_TOKENS - TARGET_TOKENS)
            return 1.5 * frac
        scores.append(_safe(_score)(text))
    return scores

def reward_avoid_multitoken_words(prompts, completions, gold, **kwargs):
    """Penalize reasoning that leans on words costing >1 BPE token (proper names, specific-object
    plurals, etc. -- see the empirical word-cost analysis: ~5% of total token cost in this corpus
    comes from ~300 multi-token words). Only pays out on CORRECT answers, same anti-gaming pattern
    as reward_token_efficiency."""
    scores = []
    for c, g in zip(completions, gold):
        text = c if isinstance(c, str) else c[0]["content"]
        def _score(t, g=g):
            pred = extract_answer_number(t)
            if pred is None or pred != g:
                return 0.0
            reasoning = extract_reasoning_span(t)
            words = WORD_RE.findall(reasoning)
            if not words:
                return 0.0
            extra_tokens = sum(word_token_cost(w) for w in words)
            return max(0.0, 1.0 - 0.2 * extra_tokens)
        scores.append(_safe(_score)(text))
    return scores

_step_counter = {"n": 0}

def reward_logger(prompts, completions, gold, **kwargs):
    _step_counter["n"] += 1
    try:
        if _step_counter["n"] % 4 == 0:
            text = completions[0] if isinstance(completions[0], str) else completions[0][0]["content"]
            reasoning = extract_reasoning_span(text)
            n_tokens = len(_COUNT_TOKENIZER.encode(reasoning, add_special_tokens=False))
            pred = extract_answer_number(text)
            rec = {"call": _step_counter["n"], "gold": gold[0], "pred": pred,
                   "correct": pred == gold[0], "reasoning_tokens": n_tokens,
                   "reasoning_excerpt": reasoning[:300], "full_len": len(text)}
            _log_f.write(json.dumps(rec) + "\n")
            _log_f.flush()
    except Exception as e:
        print(f"[reward-safety] reward_logger caught {type(e).__name__}: {e}")
    return [0.0 for _ in completions]

if __name__ == "__main__":
    random.seed(0)
    _train_limit = os.environ.get("TRAIN_LIMIT", "")
    train_rows = load_gsm8k_jsonl(os.path.join(DATA_DIR, "gsm8k_train.jsonl"),
                                   limit=int(_train_limit) if _train_limit else None)
    random.shuffle(train_rows)
    train_ds = build_dataset(train_rows)
    print(f"PHASE={PHASE} train_examples={len(train_ds)} max_steps={MAX_STEPS} "
          f"target_tokens={TARGET_TOKENS} output_dir={OUTPUT_DIR}")

    lora_r = int(os.environ.get("LORA_R", "32"))
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    reward_funcs = [reward_format, reward_correctness, reward_token_efficiency,
                     reward_avoid_multitoken_words, reward_logger]
    reward_weights = [1.0, 1.0, 1.0, 1.0, 0.0]

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.environ.get("BATCH", "8")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
        num_generations=int(os.environ.get("NUM_GEN", "12")),
        max_completion_length=512,
        learning_rate=float(os.environ.get("LR", "1.5e-5")),
        beta=float(os.environ.get("BETA", "0.06")),
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        reward_weights=reward_weights,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    if RESUME_FROM:
        print(f"Loading adapter weights from {RESUME_FROM} to resume into phase {PHASE}")
        trainer.model.load_adapter(RESUME_FROM, adapter_name="default", is_trainable=True)

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"DONE. Saved to {OUTPUT_DIR}")
