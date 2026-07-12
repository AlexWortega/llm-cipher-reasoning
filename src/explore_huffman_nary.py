import json, os, re, heapq, collections

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "gsm8k_train.jsonl")
ALPHABET_PATH = os.path.join(ROOT_DIR, "results", "explore_single_token_syms.json")
OUT_PATH = os.path.join(ROOT_DIR, "results", "explore_huffman_nary_samples.json")

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

def build_freq_table(texts):
    freq = collections.Counter()
    for t in texts:
        for w in WORD_RE.findall(t.lower()):
            freq[w] += 1
    return freq

class Node:
    __slots__ = ("freq", "sym", "children")
    def __init__(self, freq, sym=None, children=None):
        self.freq = freq; self.sym = sym; self.children = children or []
    def __lt__(self, other):
        return self.freq < other.freq

def build_nary_huffman(freq, radix, vocab_size=500):
    items = freq.most_common(vocab_size)
    heap = [Node(wt, sym) for sym, wt in items]
    # pad with zero-weight dummy nodes so (n-1) | (len-1)
    n = radix
    while (len(heap) - 1) % (n - 1) != 0:
        heap.append(Node(0, sym=None))
    heapq.heapify(heap)
    while len(heap) > 1:
        chosen = [heapq.heappop(heap) for _ in range(min(n, len(heap)))]
        heapq.heappush(heap, Node(sum(c.freq for c in chosen), children=chosen))
    return heap[0]

def codes_from_tree(root, alphabet):
    codes = {}
    def walk(node, prefix):
        if not node.children:
            if node.sym is not None:
                codes[node.sym] = prefix
            return
        for i, ch in enumerate(node.children):
            walk(ch, prefix + alphabet[i])
    walk(root, "")
    return codes

def encode_reasoning(text, codes):
    out = []
    pos = 0
    for m in WORD_RE.finditer(text):
        out.append(text[pos:m.start()])
        w = m.group(0)
        lw = w.lower()
        if lw in codes:
            out.append(codes[lw])
        else:
            out.append(w)  # OOV: keep literal word (no bracket overhead this time)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)

if __name__ == "__main__":
    texts = load_reasoning_texts(DATA_PATH)
    freq = build_freq_table(texts)

    # alphabet: single-BPE-token CJK glyphs, used as an n-ary Huffman coding alphabet instead of
    # bits, so each "digit" of the code is guaranteed to be exactly 1 token under Qwen's tokenizer
    alphabet = json.load(open(ALPHABET_PATH))
    radix = len(alphabet)
    print("radix:", radix)

    tree = build_nary_huffman(freq, radix, vocab_size=500)
    codes = codes_from_tree(tree, alphabet)
    lens = sorted(len(c) for c in codes.values())
    print("code length (symbols) range:", lens[0], "-", lens[-1], "mean:", sum(lens)/len(lens))
    print("words with 1-symbol code:", sum(1 for l in lens if l==1), "/", len(lens))

    import random
    random.seed(0)
    samples = random.sample(texts, 5)
    encoded_out = []
    for s in samples:
        enc = encode_reasoning(s, codes)
        encoded_out.append({"plain": s, "huffman": enc})
    json.dump(encoded_out, open(OUT_PATH, "w"), indent=2)
    print(f"wrote {OUT_PATH}")
