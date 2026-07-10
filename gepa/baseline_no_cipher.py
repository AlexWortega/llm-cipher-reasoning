import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cipher_utils import extract_final_number, load_gsm8k
from gepa_cipher_reasoning import call_task_lm, load_data, budget

PLAIN_PROMPT = """You are solving grade-school math word problems.

Think through the problem step by step in plain English, then give the final numeric answer on \
its own line, formatted exactly as:
#### <number>"""

if __name__ == "__main__":
    trainset, valset, testset = load_data()
    print(f"Running PLAIN (no-cipher) baseline on {len(testset)} held-out test examples, "
          f"model={os.environ.get('TASK_MODEL','moonshotai/kimi-k2-thinking')}")

    results = []
    def run_one(ex):
        r = call_task_lm(PLAIN_PROMPT, ex["question"], max_tokens=2000, temperature=0.3)
        pred = extract_final_number(r["content"]) if r and r.get("content") else None
        correct = pred is not None and pred == ex["gold"]
        total_len = len(r.get("reasoning","")) + len(r.get("content","")) if r else 0
        return {"question": ex["question"], "gold": ex["gold"], "pred": pred,
                "correct": correct, "length_chars": total_len}

    with ThreadPoolExecutor(max_workers=20) as ex_pool:
        futs = [ex_pool.submit(run_one, ex) for ex in testset]
        for fut in as_completed(futs):
            results.append(fut.result())

    n = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    avg_len = sum(r["length_chars"] for r in results) / n
    print(f"\nPLAIN (no cipher) baseline: accuracy={correct_n/n:.2%} ({correct_n}/{n}), "
          f"avg reasoning+content length={avg_len:.0f} chars")
    print(f"task-LM spend for this baseline: ${budget.spent:.4f} over {budget.calls} calls")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    with open(os.path.join(results_dir, "baseline_no_cipher_result.json"), "w") as f:
        json.dump({"accuracy": correct_n/n, "n": n, "correct_n": correct_n,
                    "avg_length_chars": avg_len, "spend": budget.spent, "calls": budget.calls,
                    "results": results}, f, indent=2)
