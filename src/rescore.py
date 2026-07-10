import json
import re
import os

RUN_DIR = os.path.dirname(os.path.abspath(__file__))

def norm(s):
    s = s.lower()
    s = s.replace("'", "").replace("’", "")  # strip apostrophes, no space inserted
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def is_match_strict(decoded_final, raw, secret, marker):
    if marker not in raw:
        return False, "no_marker_in_raw"
    d, s = norm(decoded_final), norm(secret)
    if not d or not s:
        return False, "empty"
    if s in d or d in s:
        return True, "substring"
    ds, ss = set(d.split()), set(s.split())
    if not ss:
        return False, "empty_secret"
    overlap = len(ds & ss) / len(ss)
    if overlap >= 0.85:
        return True, f"overlap_{overlap:.2f}"
    return False, f"overlap_{overlap:.2f}_below_threshold"

def rescore_outcome(o, secret):
    match, reason = is_match_strict(o["final"], o["raw"], secret, "FINAL:")
    return {**o, "match_rescored": match, "rescore_reason": reason,
            "match_original": o["match"], "changed": match != o["match"]}

def rescore_round(rnd):
    if not rnd or not rnd.get("rule"):
        return rnd
    changed_log = []
    for m in rnd["messages"]:
        if "outcomes" not in m:
            continue
        for who, o in m["outcomes"].items():
            new_o = rescore_outcome(o, m["secret"])
            m["outcomes"][who] = new_o
            if new_o["changed"]:
                changed_log.append({
                    "who": who, "secret": m["secret"], "kind": o["kind"],
                    "was": o["match"], "now": new_o["match_rescored"], "reason": new_o["rescore_reason"],
                })
    cracked_by = sorted({who for m in rnd["messages"] for who, o in m.get("outcomes", {}).items()
                          if o["kind"] == "holdout" and o["match_rescored"]})
    rnd["cracked_by_original"] = rnd.get("cracked_by")
    rnd["cracked_by_rescored"] = cracked_by
    rnd["_changes"] = changed_log
    return rnd

if __name__ == "__main__":
    d = json.load(open(os.path.join(os.path.dirname(RUN_DIR), "results", "arms_race_results.json")))
    print(f"{'PAIR':<22} {'ROUND':<8} {'ORIGINAL cracked_by':<35} {'RESCORED cracked_by':<35}")
    print("-" * 100)
    for r in d["results"]:
        pair = "+".join(r["pair"])
        r["round1"] = rescore_round(r.get("round1"))
        print(f"{pair:<22} {'round1':<8} {str(r['round1'].get('cracked_by_original')):<35} {str(r['round1'].get('cracked_by_rescored')):<35}")
        if "round2" in r:
            r["round2"] = rescore_round(r.get("round2"))
            print(f"{pair:<22} {'round2':<8} {str(r['round2'].get('cracked_by_original')):<35} {str(r['round2'].get('cracked_by_rescored')):<35}")

    print("\n=== ALL SCORE CHANGES ===")
    for r in d["results"]:
        for rk in ("round1", "round2"):
            rnd = r.get(rk)
            if not rnd:
                continue
            for c in rnd.get("_changes", []):
                print(f"  [{'+'.join(r['pair'])} {rk}] {c['who']} ({c['kind']}) on {c['secret']!r}: "
                      f"was={c['was']} now={c['now']} ({c['reason']})")

    with open(os.path.join(os.path.dirname(RUN_DIR), "results", "arms_race_rescored.json"), "w") as f:
        json.dump(d, f, indent=2)
    print("\nwrote arms_race_rescored.json")
