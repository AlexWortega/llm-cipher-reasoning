import json, os, re, random

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

WORD_RE = re.compile(r"[A-Za-z']+")

TIER4_STOPWORDS = set(
    "the a an is was were are of to in that which so it its for on at by "
    "this these those his her their they we i you your my our he she its if but or as with from "
    "will would can could should must has have had do does did be been being "
    "also then now there here just very much many each every all both some".split()
)

def strip_words(t, stopset):
    def repl(m):
        w = m.group(0)
        return "" if w.lower() in stopset else w
    out = WORD_RE.sub(repl, t)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" ([,.])", r"\1", out)
    return out.strip()

def gsm8k_gold(answer_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not m:
        return None
    s = m.group(1).replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def build_reasoning_target(answer_field):
    reasoning = re.sub(r"####.*$", "", answer_field, flags=re.DOTALL).strip()
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)  # drop calculator annotations, keep the surrounding arithmetic text
    stripped = strip_words(reasoning, TIER4_STOPWORDS)
    stripped = re.sub(r"\n+", " ", stripped).strip()
    stripped = re.sub(r"^\s*,\s*", "", stripped)          # stray leading comma from a removed leading word
    stripped = re.sub(r"\.\s*,\s*", ". ", stripped)       # ". , " -> ". " left over from a removed sentence-starter
    stripped = re.sub(r"  +", " ", stripped)
    return stripped.strip()

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

def load_gsm8k(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gold = gsm8k_gold(d["answer"])
            if gold is None:
                continue
            target_reasoning = build_reasoning_target(d["answer"])
            if not target_reasoning:
                continue
            rows.append({
                "question": d["question"],
                "gold": gold,
                "reasoning": target_reasoning,
            })
    return rows

if __name__ == "__main__":
    random.seed(0)
    rows = load_gsm8k(os.path.join(DATA_DIR, "gsm8k_train.jsonl"))
    random.shuffle(rows)
    print(f"built {len(rows)} SFT examples")

    out_path = os.path.join(DATA_DIR, "sft_tier4_train.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": f"<reasoning>\n{r['reasoning']}\n</reasoning>\n<answer>\n{r['gold']}\n</answer>"},
            ]
            f.write(json.dumps({"messages": messages}) + "\n")
    print(f"wrote {out_path}")

    # quick sample print
    for r in rows[:2]:
        print("=" * 40)
        print("Q:", r["question"][:150])
        print("REASONING:", r["reasoning"][:200])
        print("GOLD:", r["gold"])
