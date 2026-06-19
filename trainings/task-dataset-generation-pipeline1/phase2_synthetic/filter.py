"""Phase 2, quality gate 2: decontaminate, dedup, balance, and assemble the training set.

Takes the judge's kept traces and turns them into the final training set:
  1. keep only judged rows with keep=True and a valid label.
  2. decontaminate by issue id against the sacred gold ids (data/gold_ids.json), and
     drop near-duplicate serialized contexts (char-shingle Jaccard).
  3. balance across the two classes AND across trace-length buckets, so neither the
     majority class nor one reasoning shape dominates.
  4. assemble train_synth.jsonl = ALL seed rows + kept synthetic (keeping the real
     seeds in the mix guards against mode collapse).

Reads  : data/judged.jsonl, data/seed.jsonl, data/gold_ids.json
Writes : data/train_synth.jsonl  (messages format, ready for Phase 3 / rehearsal mix)

Run from the track root:
    ../../.venv/bin/python phase2_synthetic/filter.py
"""

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common

JUDGED = f"{common.DATA}/judged.jsonl"
OUT_FILE = f"{common.DATA}/train_synth.jsonl"
DUP_SIM = 0.85   # near-dup contexts above this Jaccard are redundant, drop


def near(shingles, pool, threshold):
    return any(common.jaccard(shingles, s) >= threshold for s in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-class", type=int, default=0,
                    help="optional max kept synthetic rows per class (0 = no cap)")
    args = ap.parse_args()

    labels = common.load_labels()
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    judged = common.read_jsonl(JUDGED)
    gold_ids = set(json.load(open(f"{common.DATA}/gold_ids.json")))

    kept_rows, kept_sh = [], []
    per_class, per_bucket = Counter(), Counter()
    dropped = {"judge": 0, "bad_label": 0, "gold_id": 0, "dup_near": 0, "class_cap": 0}

    for r in judged:
        if not r.get("keep"):
            dropped["judge"] += 1
            continue
        label = r["label"]
        if label not in labels:
            dropped["bad_label"] += 1
            continue
        if str(r["id"]) in gold_ids:        # decontaminate by issue id
            dropped["gold_id"] += 1
            continue
        sh = common.char_shingles(r["context"])
        if near(sh, kept_sh, DUP_SIM):
            dropped["dup_near"] += 1
            continue
        if args.cap_per_class and per_class[label] >= args.cap_per_class:
            dropped["class_cap"] += 1
            continue
        kept_rows.append(common.build_row(r["context"], r["reasoning"], label, labels))
        kept_sh.append(sh)
        per_class[label] += 1
        per_bucket[r.get("trace_len_bucket", "?")] += 1

    final = list(seeds) + kept_rows
    common.write_jsonl(OUT_FILE, final)

    print(f"judged in       : {len(judged)}")
    print(f"kept synthetic  : {len(kept_rows)}  per-class={dict(per_class)}")
    print(f"trace-length mix: {dict(per_bucket)}")
    print(f"dropped         : {dropped}")
    print(f"final train rows: {len(final)}  (= {len(seeds)} seeds + {len(kept_rows)} synthetic)")
    thin = [l for l in labels if per_class[l] < 5]
    if thin:
        print(f"WARNING thin classes (<5 synthetic): {thin}")
    print(f"\nWrote {OUT_FILE}. Next: mix_rehearsal.py (or train on this directly).")


if __name__ == "__main__":
    main()
