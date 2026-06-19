"""Phase 1 for Track A: define the task and build the seed + gold sets.

What this produces (all under ./data):
  labels.txt   the 77-intent closed set, fixed order  (the output vocabulary)
  gold.jsonl   the SACRED held-out evaluation set, reserved first
  seed.jsonl   a small verified training seed, in the exact task shape

Disciplines enforced here (so Phase 2+ can trust them):
  - Gold is reserved BEFORE anything else and is drawn from banking77's TEST
    split. The seed is drawn from the TRAIN split. The two splits are disjoint
    upstream, so nothing in gold can leak into training.
  - Both sets are stratified (an equal number per intent) so no label dominates.
  - We assert there is zero normalized-query overlap between seed and gold.
  - Everything is deterministic (fixed SEED) so the run reproduces.

Run from the track_a_banking77 folder with the project venv (plain `python` is
the wrong interpreter on this machine):
    ../../../.venv/bin/python phase1_seed/build_seed.py
"""

import os
import random
import sys

from datasets import load_dataset

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

# ------------------------------- knobs --------------------------------
SEEDS_PER_INTENT = 5    # small on purpose: leaves headroom for synthetic data
GOLD_PER_INTENT  = 20   # the held-out gold set, sacred
SEED = 13
# ----------------------------------------------------------------------


def stratified(rows, key_fn, n_per_key, rng):
    """Pick n_per_key items for each key, sampled without replacement."""
    buckets = {}
    for r in rows:
        buckets.setdefault(key_fn(r), []).append(r)
    picked = []
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        picked.extend(pool[:n_per_key])
    return picked


def main():
    rng = random.Random(SEED)

    # banking77 still ships a legacy loader script, which new `datasets`
    # refuses to run. The Hub auto-converts every dataset to a parquet branch
    # (refs/convert/parquet) that has no script and keeps the ClassLabel names,
    # so we load that revision instead.
    ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
    # banking77 ships the label names in the ClassLabel feature.
    label_names = ds["train"].features["label"].names
    with open(common.LABELS_PATH, "w") as f:
        for name in label_names:
            f.write(name + "\n")
    labels = common.load_labels()
    assert len(labels) == 77, f"expected 77 intents, got {len(labels)}"

    def to_record(ex):
        return {"text": ex["text"].strip(), "intent": label_names[ex["label"]]}

    train = [to_record(ex) for ex in ds["train"]]
    test  = [to_record(ex) for ex in ds["test"]]

    # 1) Reserve gold FIRST, from the test split.
    gold_recs = stratified(test, lambda r: r["intent"], GOLD_PER_INTENT, rng)
    gold_norm = {common.normalize(r["text"]) for r in gold_recs}

    # 2) Clean the train pool BEFORE sampling the seed:
    #    - decontaminate: drop any train query that also appears in gold
    #      (banking77's train/test splits do share some identical queries)
    #    - dedup: keep only the first occurrence of each normalized query
    seen, dropped_leak, dropped_dup = set(), 0, 0
    clean_train = []
    for r in train:
        n = common.normalize(r["text"])
        if n in gold_norm:
            dropped_leak += 1
            continue
        if n in seen:
            dropped_dup += 1
            continue
        seen.add(n)
        clean_train.append(r)

    # 3) Sample the small, balanced seed from the cleaned pool.
    seed_recs = stratified(clean_train, lambda r: r["intent"], SEEDS_PER_INTENT, rng)

    # 4) Leak guard: after decontamination this must be exactly zero.
    leaks = [r for r in seed_recs if common.normalize(r["text"]) in gold_norm]
    assert not leaks, f"{len(leaks)} seed rows still overlap the gold set"

    seed_rows = [common.build_row(r["text"], r["intent"], labels) for r in seed_recs]
    gold_rows = [common.build_row(r["text"], r["intent"], labels) for r in gold_recs]

    common.write_jsonl(f"{common.DATA}/seed.jsonl", seed_rows)
    common.write_jsonl(f"{common.DATA}/gold.jsonl", gold_rows)

    print(f"labels : {len(labels)} intents -> data/labels.txt")
    print(f"seed   : {len(seed_rows)} rows ({SEEDS_PER_INTENT}/intent) -> data/seed.jsonl")
    print(f"gold   : {len(gold_rows)} rows ({GOLD_PER_INTENT}/intent) -> data/gold.jsonl")
    print(f"cleaned train pool: dropped {dropped_leak} gold-leak + {dropped_dup} dup queries")
    print(f"leak check: 0 overlapping queries between seed and gold")
    print("\nexample seed row:")
    ex = seed_rows[0]
    print("  user     :", common.user_of(ex))
    print("  assistant:", common.assistant_of(ex))


if __name__ == "__main__":
    main()
