import json, os, re, random

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "gsm8k_train.jsonl")
OUT_PATH = os.path.join(ROOT_DIR, "results", "explore_stopword_tiers_samples.json")

WORD_RE = re.compile(r"[A-Za-z']+")

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
random.seed(1)
samples = random.sample(texts, 150)

TIER1 = set("the a an is was were are of to in that which so it its for on at by".split())
TIER2 = TIER1 | set("this these those his her their they we i you your my our he she its if but or as with from".split())
TIER3 = TIER2 | set("will would can could should must has have had do does did be been being".split())
TIER4 = TIER3 | set("also then now there here just very much many each every all both some".split())

def strip_words(t, stopset):
    def repl(m):
        w = m.group(0)
        return "" if w.lower() in stopset else w
    out = WORD_RE.sub(repl, t)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" ([,.])", r"\1", out)
    return out.strip()

variants = {
    "plain": lambda t: t,
    "tier1_basic": lambda t: strip_words(t, TIER1),
    "tier2_pronouns": lambda t: strip_words(t, TIER2),
    "tier3_aux_verbs": lambda t: strip_words(t, TIER3),
    "tier4_aggressive": lambda t: strip_words(t, TIER4),
}

if __name__ == "__main__":
    out = {name: [fn(s) for s in samples] for name, fn in variants.items()}
    json.dump(out, open(OUT_PATH, "w"), indent=2)

    print("=== sample of tier4 (most aggressive) ===")
    for i in range(3):
        print("PLAIN:", samples[i][:200])
        print("TIER4:", out["tier4_aggressive"][i][:200])
        print()
    print(f"wrote {OUT_PATH}")
