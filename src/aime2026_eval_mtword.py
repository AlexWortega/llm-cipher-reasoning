import json
import os
import re
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_train_efficient import SYSTEM_PROMPT, extract_reasoning_span

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT_DIR, "checkpoints"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.join(CKPT_DIR, "ckpt_phase_eff_mtword"))
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
LABEL = os.environ.get("LABEL", "trained")
USE_ADAPTER = os.environ.get("USE_ADAPTER", "1") == "1"

def extract_final_int(text):
    m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\\boxed\{(-?\d+)\}", text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def generate(model, tokenizer, question, max_new_tokens=2048, temperature=0.3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            top_p=0.9, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    ds = load_dataset("MathArena/aime_2026")["train"]
    print(f"AIME 2026: {len(ds)} problems, label={LABEL}, use_adapter={USE_ADAPTER}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    if USE_ADAPTER:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.eval()

    results = []
    for i, row in enumerate(ds):
        gold = int(row["answer"])
        text = generate(model, tokenizer, row["problem"])
        pred = extract_final_int(text)
        correct = pred is not None and pred == gold
        reasoning = extract_reasoning_span(text)
        results.append({
            "problem_idx": row["problem_idx"], "gold": gold, "pred": pred,
            "correct": correct, "text": text, "reasoning_only": reasoning,
        })
        print(f"[{i+1}/{len(ds)}] gold={gold} pred={pred} correct={correct}")

    n = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    acc = correct_n / n
    summary = {"label": LABEL, "model": MODEL_PATH + (f" + {ADAPTER_PATH}" if USE_ADAPTER else " (base)"),
               "accuracy": acc, "correct_n": correct_n, "n": n, "results": results}

    out_path = os.path.join(RESULTS_DIR, f"aime2026_mtword_{LABEL}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY === label={LABEL} accuracy={acc:.2%} ({correct_n}/{n}) -> {out_path}")
