import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_cipher_reasoning import evaluate, load_data, SEED_PROMPT, budget

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gepa_outputs", "candidates.json")) as f:
    candidates = json.load(f)
BEST_CANDIDATE = candidates[22]["current_candidate"]

trainset, valset, testset = load_data()
print(f"Held-out test set: {len(testset)} examples\n")

def eval_on_set(prompt, dataset, label):
    scores, results = [], []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(evaluate, prompt, e): e for e in dataset}
        for fut in as_completed(futs):
            s, info = fut.result()
            scores.append(s)
            results.append(info)
    correct_n = sum(1 for r in results if r.get("model_final_answer") == r.get("gold_answer"))
    acc = correct_n / len(dataset)
    avg_score = sum(scores) / len(scores)
    print(f"{label}: accuracy={acc:.2%} ({correct_n}/{len(dataset)}) avg_score={avg_score:.3f}")
    return {"accuracy": acc, "correct_n": correct_n, "n": len(dataset), "avg_score": avg_score,
             "results": results}

print("=== SEED prompt on held-out test ===")
seed_res = eval_on_set(SEED_PROMPT, testset, "SEED")
print("=== BEST (GEPA-optimized, candidate 22) on held-out test ===")
best_res = eval_on_set(BEST_CANDIDATE, testset, "OPTIMIZED")

print(f"\nSpend for this final eval: ${budget.spent:.4f} over {budget.calls} calls")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
with open(os.path.join(RESULTS_DIR, "gepa_final_eval_result.json"), "w") as f:
    json.dump({"seed_prompt": SEED_PROMPT, "best_candidate": BEST_CANDIDATE,
                "seed_result": seed_res, "best_result": best_res,
                "final_eval_spend": budget.spent}, f, indent=2)
