import re
import json

DICT_PATH = "/usr/share/dict/words"

def load_dict():
    words = set()
    with open(DICT_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w.isalpha():
                words.add(w)
    return words

ENGLISH_WORDS = load_dict()

WORD_RE = re.compile(r"[A-Za-z]+")

def word_tokens(text):
    return WORD_RE.findall(text)

def cipher_encode_word(w):
    return w[::-1]

def is_reversed_form(token):
    """True if token looks like a word deliberately reversed under our cipher
    (its reverse is a real dictionary word, but the token itself usually isn't)."""
    lw = token.lower()
    if len(lw) < 3:
        return None  # too short to judge (a/I/an/etc. are invariant or ambiguous)
    rev = lw[::-1]
    rev_is_word = rev in ENGLISH_WORDS
    self_is_word = lw in ENGLISH_WORDS
    if lw == rev:
        return None  # palindromes are ambiguous, don't count either way
    if rev_is_word and not self_is_word:
        return True
    if self_is_word and not rev_is_word:
        return False
    return None  # ambiguous (both or neither are real words) -- skip

def cipher_adherence_score(text):
    """Fraction of judgeable words in `text` that appear to be cipher-encoded
    (reversed). Returns (score in [0,1] or None if no judgeable tokens, n_judged)."""
    tokens = word_tokens(text)
    judged = [is_reversed_form(t) for t in tokens]
    judged = [j for j in judged if j is not None]
    if not judged:
        return None, 0
    return sum(judged) / len(judged), len(judged)

def extract_final_number(text):
    """Extract the final numeric answer GSM8K-style: prefer '#### N' if present,
    else the last standalone number in the text."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return normalize_num(m.group(1))
    nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", text)
    if not nums:
        return None
    return normalize_num(nums[-1])

def normalize_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s

def gsm8k_gold_answer(answer_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return normalize_num(m.group(1)) if m else None

def load_gsm8k(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({"question": d["question"], "gold": gsm8k_gold_answer(d["answer"])})
            if limit and len(rows) >= limit:
                break
    return rows

if __name__ == "__main__":
    tests = [
        "Ew deen ot dnif eht rebmun fo eggs.",  # reversed english-ish
        "We need to find the number of eggs.",  # plain english
        "16 - 3 - 4 = 9 elttub selppa",
    ]
    for t in tests:
        print(t, "->", cipher_adherence_score(t))
