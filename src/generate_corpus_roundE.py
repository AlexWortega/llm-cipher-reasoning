import json
import os
import sys
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_train_efficient import SYSTEM_PROMPT, extract_reasoning_span, extract_answer_number, gsm8k_gold

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT_DIR, "checkpoints"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.join(CKPT_DIR, "ckpt_phase_eff3"))
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "400"))

def generate(model, tokenizer, question, max_new_tokens=200, temperature=0.5):
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
    random.seed(7)
    rows = []
    with open(os.path.join(DATA_DIR, "gsm8k_train.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            gold = gsm8k_gold(d["answer"])
            if gold is not None:
                rows.append({"question": d["question"], "gold": gold})
    random.shuffle(rows)
    rows = rows[:N_SAMPLES]
    print(f"Sampling {len(rows)} GSM8K train problems with {ADAPTER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    out_path = os.path.join(RESULTS_DIR, "roundE_corpus", "fresh_eff3_corpus.jsonl")
    n_correct = 0
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            text = generate(model, tokenizer, r["question"])
            reasoning = extract_reasoning_span(text)
            pred = extract_answer_number(text)
            correct = pred is not None and pred == r["gold"]
            n_correct += correct
            if reasoning:
                f.write(json.dumps({"question": r["question"], "gold": r["gold"], "pred": pred,
                                     "correct": correct, "reasoning_excerpt": reasoning}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(rows)}] running_acc={n_correct/(i+1):.2%}")
    print(f"DONE. accuracy={n_correct/len(rows):.2%} -> {out_path}")
