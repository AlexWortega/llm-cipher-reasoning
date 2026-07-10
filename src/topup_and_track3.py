import json
import os
from cipher_experiment import (
    call, MODELS, ENCODE_PROMPT_INVENTED, DECODE_PROMPT, REPLY_PROMPT, COST, RUN_DIR
)

SECRET_2 = "Meet at the old oak tree tomorrow."

results = {"track2_topup": [], "track3": []}

print("=== TRACK 2 TOP-UP: opus46 and gpt55 encode with bigger budget ===")
for enc in ["opus46", "gpt55"]:
    prompt = ENCODE_PROMPT_INVENTED.format(secret=SECRET_2)
    ct = call(enc, prompt, max_tokens=1500, temperature=0.9)
    print(f"enc={enc} ciphertext={ct!r}")
    row = {"encoder": enc, "ciphertext": ct, "decodes": []}
    for dec in MODELS:
        if dec == enc:
            continue
        dprompt = DECODE_PROMPT.format(ciphertext=ct)
        decoded = call(dec, dprompt, max_tokens=1200, temperature=0.3)
        print(f"  dec={dec} -> {decoded!r}")
        row["decodes"].append({"decoder": dec, "decoded": decoded})
    results["track2_topup"].append(row)

print("\n=== TRACK 3: bidirectional reply test on 2 clean successful pairs ===")
# Pair A: track2 gemini31 (alphabetize-letters cipher) -> opus48 decoded successfully cold
# Pair B: track1 gemini31 (ROT13) -> gpt55 decoded successfully cold
pairs = [
    {
        "label": "track2_gemini_to_opus48",
        "encoder": "gemini31",
        "decoder": "opus48",
        "secret": SECRET_2,
        "ciphertext": "eemt ta eht dlo ako eert mooorrtw",
    },
    {
        "label": "track1_gemini_to_gpt55",
        "encoder": "gemini31",
        "decoder": "gpt55",
        "secret": "The vault opens at midnight.",
        "ciphertext": "Gur inhyg bcraf ng zvqavtug.",
    },
]

for p in pairs:
    enc, dec, secret, ct = p["encoder"], p["decoder"], p["secret"], p["ciphertext"]
    rprompt = REPLY_PROMPT.format(secret=secret, ciphertext=ct)
    reply_ct = call(dec, rprompt, max_tokens=1200, temperature=0.9)
    print(f"[{p['label']}] reply author={dec} reply_ciphertext={reply_ct!r}")
    dprompt = DECODE_PROMPT.format(ciphertext=reply_ct)
    decoded_reply = call(enc, dprompt, max_tokens=1200, temperature=0.3)
    print(f"[{p['label']}] reply decoded by original encoder={enc} -> {decoded_reply!r}")
    results["track3"].append({
        "label": p["label"], "reply_author": dec, "reply_ciphertext": reply_ct,
        "reply_decoder": enc, "reply_decoded": decoded_reply,
    })

results["cost_tokens_this_script"] = COST
with open(os.path.join(os.path.dirname(RUN_DIR), "results", "topup_track3_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nToken usage this script:", COST)
