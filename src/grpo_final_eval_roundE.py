import json
import os
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_train_efficient import (
    extract_answer_number, extract_reasoning_span, gsm8k_gold, SYSTEM_PROMPT,
)

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT_DIR, "checkpoints"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
ORIG_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
SFT_MODEL_DIR = os.environ.get("SFT_MODEL_DIR", os.path.join(CKPT_DIR, "qwen3_vocab_ext_sft"))
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.join(CKPT_DIR, "ckpt_roundE_supertoken"))
SUPERTOKENS_PATH = os.environ.get("SUPERTOKENS_PATH", os.path.join(RESULTS_DIR, "roundE_supertokens.json"))
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
N_EVAL = int(os.environ.get("N_EVAL", "70"))

PLAIN_PROMPT = """You are solving grade-school math word problems.

Solve the problem, then respond in exactly this structure:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer, digits only
</answer>"""

def load_eval_set(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gold = gsm8k_gold(d["answer"])
            if gold is not None:
                rows.append({"question": d["question"], "gold": gold})
    random.Random(1).shuffle(rows)
    return rows[:n]

def generate(model, tokenizer, system_prompt, question, max_new_tokens=512, temperature=0.3):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=0.9, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text

def supertoken_usage(reasoning, supertoken_strs):
    count = sum(reasoning.count(st) for st in supertoken_strs)
    used = count > 0
    return used, count

def run_condition(model, tokenizer, count_tokenizer, system_prompt, eval_set, label,
                   save_qual=False, supertoken_strs=None):
    results = []
    qual_samples = []
    for i, ex in enumerate(eval_set):
        text = generate(model, tokenizer, system_prompt, ex["question"])
        pred = extract_answer_number(text)
        correct = pred is not None and pred == ex["gold"]
        reasoning = extract_reasoning_span(text)
        n_tokens = len(count_tokenizer.encode(reasoning, add_special_tokens=False))
        rec = {"question": ex["question"], "gold": ex["gold"], "pred": pred, "correct": correct,
               "reasoning_tokens": n_tokens}
        if supertoken_strs is not None:
            used, count = supertoken_usage(reasoning, supertoken_strs)
            rec["supertoken_used"] = used
            rec["supertoken_count"] = count
        results.append(rec)
        if save_qual and i < 15:
            qual = dict(rec)
            qual["full_reasoning"] = reasoning
            qual["full_text"] = text
            qual_samples.append(qual)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(eval_set)} done")
    n = len(results)
    acc = sum(1 for r in results if r["correct"]) / n
    mean_tokens = sum(r["reasoning_tokens"] for r in results) / n
    summary = {"label": label, "accuracy": acc, "mean_reasoning_tokens": mean_tokens, "n": n,
               "results": results}
    msg = f"{label}: accuracy={acc:.2%} mean_reasoning_tokens={mean_tokens:.1f}"
    if supertoken_strs is not None:
        usage_rate = sum(1 for r in results if r["supertoken_used"]) / n
        mean_count = sum(r["supertoken_count"] for r in results) / n
        summary["supertoken_usage_rate"] = usage_rate
        summary["supertoken_mean_count"] = mean_count
        msg += f" supertoken_usage_rate={usage_rate:.2%} supertoken_mean_count={mean_count:.2f}"
    print(msg)
    return summary, qual_samples

if __name__ == "__main__":
    eval_set = load_eval_set(os.path.join(DATA_DIR, "gsm8k_test.jsonl"), N_EVAL)
    print(f"Eval set size: {len(eval_set)}")

    supertokens = json.load(open(SUPERTOKENS_PATH))["supertokens"]
    supertoken_strs = [st["new_token"] for st in supertokens]
    print(f"Tracking usage of {len(supertoken_strs)} supertokens: {supertoken_strs}")

    orig_tokenizer = AutoTokenizer.from_pretrained(ORIG_MODEL_PATH, cache_dir=HF_CACHE)
    if orig_tokenizer.pad_token_id is None:
        orig_tokenizer.pad_token = orig_tokenizer.eos_token
    orig_model = AutoModelForCausalLM.from_pretrained(
        ORIG_MODEL_PATH, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    orig_model.eval()

    print("\n=== Condition (a): plain baseline, original base model, no conciseness instruction ===")
    res_a, _ = run_condition(orig_model, orig_tokenizer, orig_tokenizer, PLAIN_PROMPT, eval_set, "plain_baseline")

    print("\n=== Condition (b): concise-notation prompt, original base model, no training ===")
    res_b, _ = run_condition(orig_model, orig_tokenizer, orig_tokenizer, SYSTEM_PROMPT, eval_set, "concise_prompt_only")

    del orig_model
    torch.cuda.empty_cache()

    ext_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if ext_tokenizer.pad_token_id is None:
        ext_tokenizer.pad_token = ext_tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    sft_model.eval()

    print("\n=== Condition (d): SFT-seed only (vocab-extended, no GRPO) ===")
    res_d, qual_d = run_condition(sft_model, ext_tokenizer, ext_tokenizer, SYSTEM_PROMPT, eval_set,
                                    "sft_seed_only", save_qual=True, supertoken_strs=supertoken_strs)

    print("\n=== Loading Round E GRPO-polished LoRA adapter on top of the SFT-seeded base ===")
    trained_model = PeftModel.from_pretrained(sft_model, ADAPTER_PATH)
    trained_model.eval()

    print("\n=== Condition (c): concise-notation prompt, GRPO-polished (Round E, supertoken vocab) ===")
    res_c, qual_c = run_condition(trained_model, ext_tokenizer, ext_tokenizer, SYSTEM_PROMPT, eval_set,
                                    "grpo_roundE_supertoken", save_qual=True, supertoken_strs=supertoken_strs)

    summary = {
        "eval_n": len(eval_set),
        "supertokens_tracked": supertoken_strs,
        "plain_baseline": {"accuracy": res_a["accuracy"], "mean_reasoning_tokens": res_a["mean_reasoning_tokens"]},
        "concise_prompt_only": {"accuracy": res_b["accuracy"], "mean_reasoning_tokens": res_b["mean_reasoning_tokens"]},
        "sft_seed_only": {"accuracy": res_d["accuracy"], "mean_reasoning_tokens": res_d["mean_reasoning_tokens"],
                           "supertoken_usage_rate": res_d["supertoken_usage_rate"],
                           "supertoken_mean_count": res_d["supertoken_mean_count"]},
        "grpo_roundE_supertoken": {"accuracy": res_c["accuracy"], "mean_reasoning_tokens": res_c["mean_reasoning_tokens"],
                                    "supertoken_usage_rate": res_c["supertoken_usage_rate"],
                                    "supertoken_mean_count": res_c["supertoken_mean_count"]},
        "full_results": {"a": res_a, "b": res_b, "d": res_d, "c": res_c},
    }
    with open(os.path.join(RESULTS_DIR, "grpo_final_eval_roundE_result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "qualitative_samples_roundE.json"), "w") as f:
        json.dump({"d_sft_seed_only": qual_d, "c_grpo_roundE": qual_c}, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"(a) plain baseline:        acc={res_a['accuracy']:.2%}  mean_tokens={res_a['mean_reasoning_tokens']:.1f}")
    print(f"(b) concise, no train:     acc={res_b['accuracy']:.2%}  mean_tokens={res_b['mean_reasoning_tokens']:.1f}")
    print(f"(d) SFT-seed only:         acc={res_d['accuracy']:.2%}  mean_tokens={res_d['mean_reasoning_tokens']:.1f}  "
          f"supertoken_usage={res_d['supertoken_usage_rate']:.2%}  mean_count={res_d['supertoken_mean_count']:.2f}")
    print(f"(c) GRPO(roundE, supertok): acc={res_c['accuracy']:.2%}  mean_tokens={res_c['mean_reasoning_tokens']:.1f}  "
          f"supertoken_usage={res_c['supertoken_usage_rate']:.2%}  mean_count={res_c['supertoken_mean_count']:.2f}")
