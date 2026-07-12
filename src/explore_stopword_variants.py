import json, os, re

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "gsm8k_train.jsonl")
OUT_PATH = os.path.join(ROOT_DIR, "results", "explore_stopword_variants_samples.json")

WORD_RE = re.compile(r"[A-Za-z]+")

def load_reasoning_texts(path, limit=None):
    texts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            reasoning = re.sub(r"####.*$", "", d["answer"], flags=re.DOTALL).strip()
            texts.append(reasoning)
            if limit and len(texts) >= limit:
                break
    return texts

texts = load_reasoning_texts(DATA_PATH)
import random
random.seed(0)
samples = random.sample(texts, 20)

# Variant A: stopword removal (delete cheap filler words entirely)
STOPWORDS = set("the a an is was were are of to in that which so it its for on at by".split())
def variant_stopword_strip(t):
    def repl(m):
        w = m.group(0)
        return "" if w.lower() in STOPWORDS else w
    out = WORD_RE.sub(repl, t)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" ([,.])", r"\1", out)
    return out

# Variant B: symbol substitution for connector/relation words only (small closed set, common ones)
SYMBOL_MAP = {
    "total": "Σ", "therefore": "∴", "so": "→", "equals": "=", "plus": "+", "minus": "-",
    "multiplied": "×", "times": "×", "divided": "÷", "remaining": "rem", "altogether": "Σ",
    "means": "=", "gives": "=", "left": "rem",
}
def variant_symbol_sub(t):
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        return SYMBOL_MAP.get(lw, w)
    return WORD_RE.sub(repl, t)

# Variant C: both combined
def variant_combined(t):
    return variant_symbol_sub(variant_stopword_strip(t))

# Variant D: aggressive telegram style - strip stopwords AND common linking phrases
LINKING_PHRASES = [
    r"\bin order to\b", r"\bthis means that\b", r"\bwe (can|need to|have to|know that)\b",
    r"\bit follows that\b", r"\blet's\b", r"\bnow,? \b",
]
def variant_telegram(t):
    out = t
    for pat in LINKING_PHRASES:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = variant_stopword_strip(out)
    out = re.sub(r"  +", " ", out).strip()
    return out

variants = {
    "plain": lambda t: t,
    "A_stopword_strip": variant_stopword_strip,
    "B_symbol_sub": variant_symbol_sub,
    "C_combined": variant_combined,
    "D_telegram": variant_telegram,
}

if __name__ == "__main__":
    out = {name: [] for name in variants}
    for s in samples:
        for name, fn in variants.items():
            out[name].append(fn(s))

    json.dump(out, open(OUT_PATH, "w"), indent=2)
    for name in variants:
        print(name, "example:", out[name][0][:150])
    print(f"wrote {OUT_PATH}")
