"""Phase 1, step 2: build the sacred gold set, a small verified seed, and the
training-candidate id pool, all split by issue id so nothing leaks.

Target task: binary resolution (Done vs Won't Do). Gold is reserved first and held
out by issue id; the seed is a small balanced set of real verified examples with a
short honest reasoning trace; the training pool is the (disjoint) set of ids that
gen_reasoning.py will expand into traced samples. Requires a CONFIRMED field survey
(run survey.py first) so serialization uses the agreed leakage filter.

Writes: data/labels.txt, data/gold.jsonl, data/seed.jsonl, data/gold_ids.json,
        data/train_ids.json

Run from the track root:
    ../../.venv/bin/python phase1_seed/build_seed.py
"""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common
import link
import serialize

SEED = 0
GOLD_PER_CLASS = 100
SEED_PER_CLASS = 8
TRAIN_PER_CLASS = 150


def templated_trace(linked, label):
    """A short, honest, field-grounded trace for the real seed rows (the generator
    produces the richer/varied traces later)."""
    iss = linked["issue"]
    t = iss.get("issue_type") or "ticket"
    p = (iss.get("issue_priority") or "unspecified").lower()
    turns = len(linked["snapshots"])
    total = serialize.humanize(iss.get("wf_total_time")) or "a short time"
    base = (f"This is a {p}-priority {t} handled across {turns} assignee "
            f"turn(s) over {total}.")
    if label == "Done":
        tail = ("It moved through the normal handling workflow and was actioned to "
                "completion, which is consistent with a Done outcome.")
    else:
        tail = ("The handling pattern shows it was not carried through as a normal "
                "fix, which is consistent with a Won't Do outcome.")
    return base + " " + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-per-class", type=int, default=GOLD_PER_CLASS)
    ap.add_argument("--seed-per-class", type=int, default=SEED_PER_CLASS)
    ap.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    args = ap.parse_args()

    survey = serialize.require_confirmed_survey()
    rng = random.Random(SEED)
    db = link.HelpDeskDB()
    labels = common.LABELS

    os.makedirs(common.DATA, exist_ok=True)
    with open(common.LABELS_PATH, "w") as f:
        f.write("\n".join(labels) + "\n")

    # Partition ids per class, disjointly: gold, then seed, then train pool.
    gold_rows, seed_rows = [], []
    gold_ids, train_ids = [], []
    per_class_report = {}
    for label in labels:
        ids = db.ids_with_resolution(label)
        rng.shuffle(ids)
        g = ids[:args.gold_per_class]
        s = ids[args.gold_per_class:args.gold_per_class + args.seed_per_class]
        tr = ids[args.gold_per_class + args.seed_per_class:
                 args.gold_per_class + args.seed_per_class + args.train_per_class]
        per_class_report[label] = {"available": len(ids), "gold": len(g),
                                   "seed": len(s), "train": len(tr)}
        for nid in g:
            ctx = serialize.serialize_issue(db.get_issue(nid), survey)
            gold_rows.append(common.build_eval_row(ctx, label, nid, labels))
            gold_ids.append(nid)
        for nid in s:
            linked = db.get_issue(nid)
            ctx = serialize.serialize_issue(linked, survey)
            seed_rows.append(common.build_row(ctx, templated_trace(linked, label), label, labels))
        train_ids.extend(nid for nid in tr)

    rng.shuffle(gold_rows)
    rng.shuffle(seed_rows)
    rng.shuffle(train_ids)

    # Gold/seed/train pools are disjoint by construction (sequential slices of the
    # shuffled per-class id list); assert as a guard.
    gset, tset = set(gold_ids), set(train_ids)
    assert gset.isdisjoint(tset), "gold and train id pools overlap"

    common.write_jsonl(f"{common.DATA}/gold.jsonl", gold_rows)
    common.write_jsonl(f"{common.DATA}/seed.jsonl", seed_rows)
    with open(f"{common.DATA}/gold_ids.json", "w") as f:
        json.dump(sorted(gset), f)
    with open(f"{common.DATA}/train_ids.json", "w") as f:
        json.dump(sorted(tset), f)

    print("per-class counts:", json.dumps(per_class_report))
    print(f"gold rows : {len(gold_rows)}  -> data/gold.jsonl")
    print(f"seed rows : {len(seed_rows)} -> data/seed.jsonl")
    print(f"train pool: {len(train_ids)} ids -> data/train_ids.json (disjoint from gold)")
    print("Next: phase2_synthetic/gen_reasoning.py")


if __name__ == "__main__":
    main()
