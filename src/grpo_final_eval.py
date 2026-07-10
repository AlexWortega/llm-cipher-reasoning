import json
import os
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_train import (
    extract_answer_number, extract_reasoning_span, cipher_adherence_score,
    normalize_num, gsm8k_gold, SYSTEM_PROMPT,
)

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.join(CKPT_DIR, "ckpt_phase1"))
HF_CACHE = os.path.join(ROOT_DIR, "hf_cache")
N_EVAL = int(os.environ.get("N_EVAL", "70"))
RESULT_TAG = os.environ.get("RESULT_TAG", "")  # optional suffix for output filenames, e.g. "_scaled"

NO_CIPHER_PROMPT = """You are solving grade-school math word problems.

Solve the problem, then respond in exactly this structure:
<reasoning>
your step-by-step reasoning here, in plain English
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

def run_condition(model, tokenizer, system_prompt, eval_set, label, save_qual=False):
    results = []
    qual_samples = []
    for i, ex in enumerate(eval_set):
        text = generate(model, tokenizer, system_prompt, ex["question"])
        pred = extract_answer_number(text)
        correct = pred is not None and pred == ex["gold"]
        reasoning = extract_reasoning_span(text)
        adherence, n_judged = cipher_adherence_score(reasoning)
        results.append({
            "question": ex["question"], "gold": ex["gold"], "pred": pred, "correct": correct,
            "adherence": adherence, "n_judged": n_judged, "reasoning_len": len(reasoning),
        })
        if save_qual and i < 15:
            qual_samples.append({
                "question": ex["question"], "gold": ex["gold"], "pred": pred, "correct": correct,
                "adherence": adherence, "n_judged": n_judged, "full_reasoning": reasoning,
                "full_text": text,
            })
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(eval_set)} done")
    n = len(results)
    acc = sum(1 for r in results if r["correct"]) / n
    judgeable = [r for r in results if r["n_judged"] >= 5]
    mean_adherence = sum(r["adherence"] for r in judgeable) / len(judgeable) if judgeable else 0.0
    print(f"{label}: accuracy={acc:.2%} mean_adherence={mean_adherence:.3f} "
          f"(over {len(judgeable)}/{n} judgeable-reasoning examples)")
    return {"label": label, "accuracy": acc, "mean_adherence": mean_adherence,
            "n": n, "n_judgeable": len(judgeable), "results": results}, qual_samples

if __name__ == "__main__":
    eval_set = load_eval_set(os.path.join(DATA_DIR, "gsm8k_test.jsonl"), N_EVAL)
    print(f"Eval set size: {len(eval_set)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    base_model.eval()

    print("\n=== Condition (a): no cipher, plain baseline ===")
    res_a, _ = run_condition(base_model, tokenizer, NO_CIPHER_PROMPT, eval_set, "no_cipher_baseline")

    print("\n=== Condition (b): cipher prompt, base model (no training) ===")
    res_b, _ = run_condition(base_model, tokenizer, SYSTEM_PROMPT, eval_set, "cipher_prompt_only")

    print("\n=== Loading Phase 1 LoRA adapter ===")
    trained_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    trained_model.eval()

    print("\n=== Condition (c): cipher prompt, GRPO-trained (Phase 1) ===")
    res_c, qual_c = run_condition(trained_model, tokenizer, SYSTEM_PROMPT, eval_set,
                                    "grpo_trained", save_qual=True)

    summary = {
        "eval_n": len(eval_set),
        "no_cipher_baseline": {"accuracy": res_a["accuracy"], "mean_adherence": res_a["mean_adherence"]},
        "cipher_prompt_only": {"accuracy": res_b["accuracy"], "mean_adherence": res_b["mean_adherence"]},
        "grpo_trained": {"accuracy": res_c["accuracy"], "mean_adherence": res_c["mean_adherence"]},
        "full_results": {"a": res_a, "b": res_b, "c": res_c},
    }
    with open(os.path.join(RESULTS_DIR, f"grpo_final_eval_result{RESULT_TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"qualitative_samples{RESULT_TAG}.json"), "w") as f:
        json.dump(qual_c, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"(a) no cipher:         acc={res_a['accuracy']:.2%}  adherence={res_a['mean_adherence']:.3f}")
    print(f"(b) cipher, no train:  acc={res_b['accuracy']:.2%}  adherence={res_b['mean_adherence']:.3f}")
    print(f"(c) cipher, GRPO:      acc={res_c['accuracy']:.2%}  adherence={res_c['mean_adherence']:.3f}")
