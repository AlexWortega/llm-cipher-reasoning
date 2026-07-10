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
DICT_PATH = os.path.join(DATA_DIR, "dict_words.txt")
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
PHASE = os.environ.get("PHASE", "0")  # "0" = warmup (correctness+format only), "1" = full reward
MAX_STEPS = int(os.environ.get("MAX_STEPS", "40" if PHASE == "0" else "260"))
RESUME_FROM = os.environ.get("RESUME_FROM", "")  # LoRA adapter dir to resume from, or ""
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_DIR, "checkpoints", f"ckpt_phase{PHASE}"))

# ---------------- dictionary + cipher adherence (ported from cipher_utils.py) ----------------
def load_dict():
    words = set()
    with open(DICT_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w.isalpha():
                words.add(w)
    return words

ENGLISH_WORDS = load_dict()
WORD_RE = re.compile(r"[A-Za-z]+")

def is_reversed_form(token):
    lw = token.lower()
    if len(lw) < 3:
        return None
    rev = lw[::-1]
    rev_is_word = rev in ENGLISH_WORDS
    self_is_word = lw in ENGLISH_WORDS
    if lw == rev:
        return None
    if rev_is_word and not self_is_word:
        return True
    if self_is_word and not rev_is_word:
        return False
    return None

def cipher_adherence_score(text):
    tokens = WORD_RE.findall(text)
    judged = [j for j in (is_reversed_form(t) for t in tokens) if j is not None]
    if not judged:
        return 0.0, 0
    return sum(judged) / len(judged), len(judged)

def normalize_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        if not (f == f) or f in (float("inf"), float("-inf")):  # NaN / inf guard
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

def has_valid_format(text):
    return bool(re.search(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL))

# ---------------- data ----------------
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

SYSTEM_PROMPT = """You are solving grade-school math word problems.

Solve the problem, then respond in exactly this structure:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer, digits only
</answer>

CRITICAL RULE for the <reasoning> section: every word inside <reasoning>...</reasoning> must be \
written in a private cipher -- reverse the order of the letters within each word (e.g. "the" -> \
"eht", "total" -> "latot"). Numbers, punctuation, and spaces stay unchanged; only reverse the \
letters of alphabetic words. The <answer> section must NOT be ciphered -- plain digits only."""

def build_dataset(rows):
    data = []
    for r in rows:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
        ]
        data.append({"prompt": prompt, "gold": r["gold"]})
    return Dataset.from_list(data)

# ---------------- reward functions ----------------
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

def reward_cipher_adherence(prompts, completions, **kwargs):
    if PHASE == "0":
        return [0.0 for _ in completions]
    scores = []
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        def _score(t):
            reasoning = extract_reasoning_span(t)
            adherence, n_judged = cipher_adherence_score(reasoning)
            # require a minimum amount of judgeable content so the model can't dodge by writing
            # a one-word "reasoning" span to trivially claim high adherence
            return adherence * 1.5 if n_judged >= 8 else adherence * 1.5 * (n_judged / 8)
        scores.append(_safe(_score)(text))
    return scores

def reward_brevity(prompts, completions, **kwargs):
    if PHASE == "0":
        return [0.0 for _ in completions]
    scores = []
    target_chars = 600
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        def _score(t):
            reasoning = extract_reasoning_span(t)
            L = len(reasoning)
            return max(0.0, 0.5 * (1.0 - L / target_chars))
        scores.append(_safe(_score)(text))
    return scores

_step_counter = {"n": 0}

def reward_logger(prompts, completions, gold, **kwargs):
    """Not a real reward (returns 0) -- just logs a sample for offline inspection."""
    _step_counter["n"] += 1
    try:
        if _step_counter["n"] % 4 == 0:  # log ~1 in 4 minibatches to keep the file small
            text = completions[0] if isinstance(completions[0], str) else completions[0][0]["content"]
            reasoning = extract_reasoning_span(text)
            adherence, n_judged = cipher_adherence_score(reasoning)
            pred = extract_answer_number(text)
            rec = {"call": _step_counter["n"], "gold": gold[0], "pred": pred,
                   "correct": pred == gold[0], "adherence": adherence, "n_judged": n_judged,
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
    print(f"PHASE={PHASE} train_examples={len(train_ds)} max_steps={MAX_STEPS} output_dir={OUTPUT_DIR}")

    lora_r = int(os.environ.get("LORA_R", "32"))
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    reward_funcs = [reward_format, reward_correctness, reward_cipher_adherence, reward_brevity, reward_logger]
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

    model_id = RESUME_FROM if RESUME_FROM else MODEL_PATH
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    if RESUME_FROM:
        from peft import PeftModel
        print(f"Loading adapter weights from {RESUME_FROM} to resume into phase {PHASE}")
        trainer.model.load_adapter(RESUME_FROM, adapter_name="default", is_trainable=True)

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"DONE. Saved to {OUTPUT_DIR}")
