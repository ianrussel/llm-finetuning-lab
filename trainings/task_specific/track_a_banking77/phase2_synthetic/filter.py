"""Phase 2, quality gate 2: dedup, decontaminate, balance, and assemble.

Takes the judge's kept candidates and turns them into the final training set:

  1. keep only judged rows with keep=True
  2. decontaminate against the SACRED gold set: drop any candidate that equals a
     gold query or is a near-duplicate of one (char-shingle Jaccard). This is the
     non-negotiable step that protects the evaluation from leaking into training.
  3. dedup: drop candidates that repeat a seed, a gold row, or an already-kept
     candidate (exact-normalized or near-duplicate), so no phrasing dominates.
  4. balance report: print per-intent counts so no label runs away with the set.
  5. assemble: train_synth.jsonl = ALL seeds + kept synthetic. Keeping the real
     seeds in the mix is the guard against mode collapse / teacher bias.

Reads  : data/judged.jsonl, data/seed.jsonl, data/gold.jsonl
Writes : data/train_synth.jsonl  (messages format, ready for Phase 3 training)

Run from the track_a_banking77 folder:
    ../../../.venv/bin/python phase2_synthetic/filter.py
"""

import os
import sys
from collections import Counter

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

JUDGED = f"{common.DATA}/judged.jsonl"
OUT_FILE = f"{common.DATA}/train_synth.jsonl"

GOLD_SIM = 0.70   # near-dup vs a gold query -> contamination, drop
DUP_SIM = 0.85    # near-dup vs something already kept -> redundant, drop


def near(shingles, pool, threshold):
    """True if `shingles` is within `threshold` Jaccard of any set in `pool`."""
    return any(common.jaccard(shingles, s) >= threshold for s in pool)


def main():
    labels = common.load_labels()
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    judged = common.read_jsonl(JUDGED)

    gold_norm = {common.normalize(common.user_of(r)) for r in gold}
    gold_sh = [common.char_shingles(common.user_of(r)) for r in gold]

    # seen = everything we already have, so synthetic must be genuinely new
    seen_norm = {common.normalize(common.user_of(r)) for r in seeds} | gold_norm
    kept_sh = [common.char_shingles(common.user_of(r)) for r in seeds]

    kept_rows, per_intent = [], Counter()
    dropped = {"judge": 0, "bad_intent": 0, "gold_exact": 0, "gold_near": 0,
               "dup_exact": 0, "dup_near": 0}

    for r in judged:
        if not r.get("keep"):
            dropped["judge"] += 1
            continue
        intent, user = r["intent"], r["user"].strip()
        if intent not in labels:
            dropped["bad_intent"] += 1
            continue
        norm = common.normalize(user)
        sh = common.char_shingles(user)
        if norm in gold_norm:
            dropped["gold_exact"] += 1
            continue
        if near(sh, gold_sh, GOLD_SIM):
            dropped["gold_near"] += 1
            continue
        if norm in seen_norm:
            dropped["dup_exact"] += 1
            continue
        if near(sh, kept_sh, DUP_SIM):
            dropped["dup_near"] += 1
            continue
        seen_norm.add(norm)
        kept_sh.append(sh)
        kept_rows.append(common.build_row(user, intent, labels))
        per_intent[intent] += 1

    # final set keeps the real seeds in the mix
    final = list(seeds) + kept_rows
    common.write_jsonl(OUT_FILE, final)

    counts = [per_intent[l] for l in labels]
    print(f"judged in        : {len(judged)}")
    print(f"kept synthetic   : {len(kept_rows)}")
    print(f"dropped          : {dropped}")
    print(f"final train rows : {len(final)}  (= {len(seeds)} seeds + {len(kept_rows)} synthetic)")
    print(f"per-intent synthetic: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f}")
    thin = [l for l in labels if per_intent[l] < 2]
    if thin:
        print(f"thin intents (<2 synthetic), watch these in eval: {thin}")
    print(f"\nWrote {OUT_FILE}. Next: Phase 3 trains the LoRA on this file.")


if __name__ == "__main__":
    main()
