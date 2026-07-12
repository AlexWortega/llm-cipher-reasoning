import json, os, re, heapq, collections

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RUN_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "gsm8k_train.jsonl")
OUT_PATH = os.path.join(ROOT_DIR, "results", "explore_huffman_binary_samples.json")

WORD_RE = re.compile(r"[A-Za-z]+")

def load_reasoning_texts(path, limit=None):
    texts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # strip the "#### N" final answer line, keep the natural-language rationale
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
    __slots__ = ("freq", "sym", "left", "right")
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq = freq; self.sym = sym; self.left = left; self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq, vocab_size=300):
    top = freq.most_common(vocab_size)
    heap = [Node(wt, sym) for sym, wt in top]
    heapq.heapify(heap)
    if len(heap) == 1:
        only = heap[0]
        return Node(only.freq, left=only, right=None)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, Node(a.freq + b.freq, left=a, right=b))
    return heap[0]

def codes_from_tree(root):
    codes = {}
    def walk(node, prefix):
        if node is None:
            return
        if node.sym is not None:
            codes[node.sym] = prefix or "0"
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")
    walk(root, "")
    return codes

def encode_reasoning(text, codes):
    # tokenize into alphabetic words vs everything else (numbers, punctuation, spaces)
    out = []
    pos = 0
    for m in WORD_RE.finditer(text):
        out.append(text[pos:m.start()])
        w = m.group(0)
        lw = w.lower()
        if lw in codes:
            out.append(codes[lw])
        else:
            out.append("[" + w + "]")  # OOV escape: keep literal word, bracket-marked
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)

if __name__ == "__main__":
    texts = load_reasoning_texts(DATA_PATH)
    freq = build_freq_table(texts)
    print("total distinct alphabetic words:", len(freq))
    print("top 15:", freq.most_common(15))
    tree = build_huffman_tree(freq, vocab_size=300)
    codes = codes_from_tree(tree)
    lens = sorted(len(c) for c in codes.values())
    print("code length range:", lens[0], "-", lens[-1], "mean:", sum(lens)/len(lens))

    # try encoding a few sample reasoning traces
    import random
    random.seed(0)
    samples = random.sample(texts, 5)
    encoded_out = []
    for s in samples:
        enc = encode_reasoning(s, codes)
        print("=" * 60)
        print("PLAIN  (%d chars):" % len(s), s[:300].replace("\n", " | "))
        print("HUFF   (%d chars):" % len(enc), enc[:300].replace("\n", " | "))
        encoded_out.append({"plain": s, "huffman": enc})
    json.dump(encoded_out, open(OUT_PATH, "w"), indent=2)
    print(f"wrote {OUT_PATH}")
