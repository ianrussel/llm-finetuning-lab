"""Phase 5, step 2: generate TARGETED synthetic data for the named weak spots.

This is the difference between iteration and just running Phase 2 again. Phase 2
generated undirected variety. Here we generate *contrastively*: error_analysis.py
named the confusable pairs, so for each weak intent A whose misses collapse into
confuser B, we take real seed messages of A and ask the local model to rewrite
them so the request is unambiguously A and would NOT be read as B. The model is
told both labels and pushed to lean on what actually separates them.

The grounding rule from Phase 2 is unchanged: the verified label A never moves.
We sharpen the phrasing of a real A message, we do not invent a different problem.
Output is the same {user, intent, method, seed} shape evolve.py uses, so it flows
straight into the SAME judge.py -> filter discipline (here, build_v2.py).

Reads  : data/error_analysis.json (targets), data/seed.jsonl (real A messages)
Writes : data/gen_targeted.jsonl

Run from the track_a_banking77 folder (Ollama must be up, same as Phase 2):
    ../../../.venv/bin/python phase5_iterate/targeted.py
"""

import argparse
import collections
import json
import os
import sys

# common.py lives one level up; sdg.py lives in phase2_synthetic (reuse the
# one Ollama connection rather than opening a second).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2_synthetic"))
import common
import sdg

OUT_FILE = f"{common.DATA}/gen_targeted.jsonl"

TARGET_SYSTEM = ("You write realistic bank customer-support messages for a labelling "
                 "dataset. You are given two support categories that get confused, and "
                 "you write messages that clearly belong to the FIRST one and could not "
                 "be mistaken for the second. You never change which category the message "
                 "is about. You output only the messages, one per line, no numbering or "
                 "extra text.")


def make(seed_msg, intent, confuser, k):
    """Ask for k messages that mean `intent` and are clearly not `confuser`,
    grounded in a real seed message so the phrasing stays in-distribution."""
    raw = sdg.chat([
        {"role": "system", "content": TARGET_SYSTEM},
        {"role": "user", "content":
            f"Category to write: {intent}\n"
            f"Category to stay clearly different from: {confuser}\n\n"
            f"Here is a real '{intent}' message for reference:\n\"{seed_msg}\"\n\n"
            f"Write {k} new, distinct, realistic customer messages that clearly mean "
            f"'{intent}' and would NOT be confused with '{confuser}'. Lean on the detail "
            f"that separates the two. One message per line."},
    ], temperature=0.9, num_predict=260)
    out = []
    for line in raw.splitlines():
        s = sdg.clean_line(line)
        if len(s.split()) >= 3:
            out.append(s)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-intent", type=int, default=3,
                    help="how many real A messages to ground on, per target intent")
    ap.add_argument("--per-seed", type=int, default=4,
                    help="how many contrastive messages to ask for per seed message")
    args = ap.parse_args()

    sdg.preflight()
    ea_path = f"{common.DATA}/error_analysis.json"
    if not os.path.exists(ea_path):
        raise SystemExit(f"no {ea_path}; run phase5_iterate/error_analysis.py first.")
    targets = json.load(open(ea_path))["targets"]
    if not targets:
        raise SystemExit("error_analysis found no targets (nothing below the F1 bar). "
                         "Lower --min-f1 there, or stop iterating.")

    # Group real seed messages by intent so each target is grounded in genuine A text.
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    by_intent = collections.defaultdict(list)
    for r in seeds:
        by_intent[common.assistant_of(r)].append(common.user_of(r))

    written = 0
    with open(OUT_FILE, "w") as f:
        for t in targets:
            intent, confuser = t["intent"], t["confuser"]
            pool = by_intent.get(intent, [])[:args.seeds_per_intent]
            for seed_msg in pool:
                for s in make(seed_msg, intent, confuser, args.per_seed):
                    f.write(json.dumps({"user": s, "intent": intent,
                                        "method": f"targeted:vs_{confuser}",
                                        "seed": seed_msg}, ensure_ascii=False) + "\n")
                    written += 1
            print(f"{intent:<34} vs {confuser:<28} total={written}".ljust(90), end="\r")
    print(f"\nWrote {written} targeted rows to {OUT_FILE}. Next: build_v2.py")


if __name__ == "__main__":
    main()
