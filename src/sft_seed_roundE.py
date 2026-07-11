import json
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT_DIR, "checkpoints"))
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", os.path.join(CKPT_DIR, "qwen3_vocab_ext_base"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(CKPT_DIR, "qwen3_vocab_ext_sft"))
CORPUS_PATH = os.environ.get("CORPUS_PATH", os.path.join(RESULTS_DIR, "roundE_corpus", "fresh_eff3_corpus.jsonl"))
SUPERTOKENS_PATH = os.environ.get("SUPERTOKENS_PATH", os.path.join(RESULTS_DIR, "roundE_supertokens.json"))

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

def build_substituters(supertokens):
    # longest surface first so e.g. "Total time" is substituted before "Total" would eat part of it
    ordered = sorted(supertokens, key=lambda st: -len(st["surface"]))
    patterns = [(re.compile(r"(?<![A-Za-z])" + re.escape(st["surface"]) + r"(?![A-Za-z])"), st["new_token"])
                for st in ordered]
    return patterns

def substitute(text, patterns):
    used = []
    for pattern, new_token in patterns:
        if pattern.search(text):
            text = pattern.sub(new_token, text)
            used.append(new_token)
    return text, used

if __name__ == "__main__":
    data = json.load(open(SUPERTOKENS_PATH))
    supertokens = data["supertokens"]
    patterns = build_substituters(supertokens)
    print(f"{len(supertokens)} supertokens loaded for substitution")

    rows = []
    n_correct = 0
    n_used_any = 0
    with open(CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("correct"):
                continue
            n_correct += 1
            reasoning, used = substitute(d["reasoning_excerpt"], patterns)
            if used:
                n_used_any += 1
            rows.append({"question": d["question"], "gold": d["gold"], "reasoning": reasoning})
    print(f"Corpus: {n_correct} correct examples, {n_used_any} had >=1 supertoken substitution "
          f"({n_used_any/max(1,n_correct):.1%})")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")

    messages_ds = []
    for r in rows:
        messages_ds.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": f"<reasoning>\n{r['reasoning']}\n</reasoning>\n<answer>\n{r['gold']}\n</answer>"},
        ]})
    train_ds = Dataset.from_list(messages_ds)
    print(f"SFT train examples: {len(train_ds)}")

    lora_r = int(os.environ.get("LORA_R", "16"))
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=os.path.join(CKPT_DIR, "sft_seed_roundE_lora"),
        per_device_train_batch_size=int(os.environ.get("BATCH", "4")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
        num_train_epochs=float(os.environ.get("EPOCHS", "3")),
        learning_rate=float(os.environ.get("LR", "1e-4")),
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=512,
        packing=False,
        assistant_only_loss=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    print("Merging LoRA into base and saving full model...")
    merged = trainer.model.merge_and_unload()
    os.makedirs(OUT_DIR, exist_ok=True)
    merged.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"DONE. Saved SFT-seeded vocab-extended model to {OUT_DIR}")
