import json
import os
import re
import sys
from collections import Counter

from transformers import AutoTokenizer

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
CORPUS_DIR = os.path.join(RESULTS_DIR, "roundE_corpus")
MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
HF_CACHE = os.environ.get("HF_CACHE", os.path.join(ROOT_DIR, "hf_cache"))

MIN_NGRAM = 2
MAX_NGRAM = 4
MIN_FREQ = int(os.environ.get("MIN_FREQ", "5"))
TOP_K = int(os.environ.get("TOP_K", "48"))

def load_reasoning_corpus():
    texts = []
    for fname in os.listdir(CORPUS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(CORPUS_DIR, fname)) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                excerpt = d.get("reasoning_excerpt")
                if excerpt:
                    texts.append(excerpt)
    for fname in ["qualitative_samples_eff3.json", "qualitative_samples_mtword.json"]:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            for d in json.load(open(p)):
                r = d.get("full_reasoning")
                if r:
                    texts.append(r)
    return texts

def mine_ngrams(tokenizer, texts):
    # document frequency (distinct source examples a gram appears in), not raw occurrence count --
    # avoids one repeated line inside a single sample, or a handful of near-duplicate templated
    # GSM8K problems, dominating the ranking with a non-generalizable gram.
    ngram_docfreq = Counter()
    ngram_display = {}
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < MIN_NGRAM:
            continue
        seen_in_doc = set()
        for n in range(MIN_NGRAM, MAX_NGRAM + 1):
            for i in range(len(ids) - n + 1):
                gram = tuple(ids[i:i + n])
                decoded = tokenizer.decode(gram)
                if not re.search(r"[A-Za-z]{2,}", decoded):
                    continue
                if re.search(r"[0-9=+\-*/<>]", decoded):
                    continue
                # drop pure formatting/whitespace-glue grams (e.g. "  \nTotal") -- keep only grams
                # whose decoded form, stripped, is a real word/phrase (reasoning content, not the
                # newline/indentation scaffolding around it)
                stripped = decoded.strip()
                if len(stripped) < 3 or not re.match(r"^[A-Za-z][A-Za-z' ]*[A-Za-z%:]?$", stripped):
                    continue
                if gram not in seen_in_doc:
                    ngram_docfreq[gram] += 1
                    seen_in_doc.add(gram)
                ngram_display[gram] = stripped
    return ngram_docfreq, ngram_display

def select_supertokens(ngram_counts, ngram_display):
    # score = frequency * tokens_saved_per_use (n-1), then greedily select non-overlapping-in-rank
    # top-K, preferring longer/more-frequent grams over their sub-grams.
    scored = []
    for gram, freq in ngram_counts.items():
        if freq < MIN_FREQ:
            continue
        n = len(gram)
        score = freq * (n - 1)
        scored.append((score, freq, n, gram, ngram_display[gram]))
    scored.sort(reverse=True)

    chosen = []
    chosen_grams = []
    for score, freq, n, gram, display in scored:
        # drop candidates that are pure sub/super-strings of an already chosen higher-score gram
        # applied to the SAME surface text (avoid selecting both "to Brandon" and "Brandon")
        is_redundant = any(
            (len(gram) < len(cg) and _is_subgram(gram, cg)) for cg in chosen_grams
        )
        if is_redundant:
            continue
        chosen.append((score, freq, n, gram, display))
        chosen_grams.append(gram)
        if len(chosen) >= TOP_K:
            break
    return chosen

def _is_subgram(small, big):
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return True
    return False

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    texts = load_reasoning_corpus()
    print(f"Corpus: {len(texts)} reasoning excerpts")

    ngram_counts, ngram_display = mine_ngrams(tokenizer, texts)
    print(f"Distinct candidate n-grams (freq>=1, alphabetic, no digits/operators): {len(ngram_counts)}")

    chosen = select_supertokens(ngram_counts, ngram_display)
    total_tokens_saveable = sum(freq * (n - 1) for score, freq, n, gram, display in chosen)
    print(f"\nSelected {len(chosen)} supertokens (MIN_FREQ={MIN_FREQ}, TOP_K={TOP_K})")
    print(f"Total token-savings potential across corpus: {total_tokens_saveable} tokens "
          f"over {len(texts)} examples")
    print("\nTop 20 by score:")
    for score, freq, n, gram, display in chosen[:20]:
        print(f"  freq={freq:3d} n={n} save/use={n-1}  {display!r}  (token_ids={gram})")

    out = {
        "corpus_size": len(texts),
        "min_freq": MIN_FREQ,
        "top_k": TOP_K,
        "supertokens": [
            {"surface": display, "token_ids": list(gram), "freq": freq, "n": n,
             "tokens_saved_per_use": n - 1, "new_token": f"<ST{i}>"}
            for i, (score, freq, n, gram, display) in enumerate(chosen)
        ],
    }
    out_path = os.path.join(RESULTS_DIR, "roundE_supertokens.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
