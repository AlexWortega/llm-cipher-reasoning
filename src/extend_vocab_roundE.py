import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))
SUPERTOKENS_PATH = os.environ.get("SUPERTOKENS_PATH", os.path.join(RESULTS_DIR, "roundE_supertokens.json"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT_DIR, "checkpoints", "qwen3_vocab_ext_base"))

if __name__ == "__main__":
    data = json.load(open(SUPERTOKENS_PATH))
    supertokens = data["supertokens"]
    print(f"Loaded {len(supertokens)} mined supertokens from {SUPERTOKENS_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16)

    new_tokens = [st["new_token"] for st in supertokens]
    added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    print(f"Added {added} new tokens to tokenizer (vocab now {len(tokenizer)})")

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    tied = model.config.tie_word_embeddings
    print(f"Resized embeddings {old_vocab_size} -> {model.get_input_embeddings().weight.shape[0]}; "
          f"tie_word_embeddings={tied}")

    in_emb = model.get_input_embeddings().weight.data
    out_emb = None if tied else model.get_output_embeddings().weight.data

    for st in supertokens:
        new_id = tokenizer.convert_tokens_to_ids(st["new_token"])
        constituent_ids = torch.tensor(st["token_ids"], dtype=torch.long)
        mean_vec = in_emb[constituent_ids].float().mean(dim=0).to(in_emb.dtype)
        in_emb[new_id] = mean_vec
        if out_emb is not None:
            out_mean_vec = out_emb[constituent_ids].float().mean(dim=0).to(out_emb.dtype)
            out_emb[new_id] = out_mean_vec

    print(f"Initialized {len(supertokens)} new embedding rows as mean of constituent tokens "
          f"(tied={tied}, so {'input+output shared' if tied else 'input and output set separately'})")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved vocab-extended base model + tokenizer to {OUT_DIR}")
