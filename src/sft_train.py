import json, os
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507")
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_DIR, "checkpoints", "ckpt_sft_tier4"))
TRAIN_LIMIT = os.environ.get("TRAIN_LIMIT", "")

def load_sft_dataset(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append(d)
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)

if __name__ == "__main__":
    train_ds = load_sft_dataset(
        os.path.join(DATA_DIR, "sft_tier4_train.jsonl"),
        limit=int(TRAIN_LIMIT) if TRAIN_LIMIT else None,
    )
    print(f"SFT train examples: {len(train_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_r = int(os.environ.get("LORA_R", "32"))
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.environ.get("BATCH", "8")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
        num_train_epochs=float(os.environ.get("EPOCHS", "2")),
        learning_rate=float(os.environ.get("LR", "1e-4")),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        packing=False,
        assistant_only_loss=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"DONE. Saved to {OUTPUT_DIR}")
