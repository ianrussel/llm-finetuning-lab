"""Phase 5, step 1: error-analyze the gold misses and NAME the pattern.

Iteration starts from evidence, not a hunch. This reads the current best model's
per-row predictions (data/preds_seed-synth.jsonl by default) and finds where it
still fails, in the exact shape targeted generation can act on:

  - per-intent precision / recall / F1, weakest first: which intents are weak,
    and whether the weakness is recall (it misses them) or precision (it over-
    fires them on neighbours).
  - confusable pairs: for each gold intent, the wrong label it is most often
    predicted as. A classifier fails as confusable PAIRS, so naming the pair is
    naming the pattern. These pairs are what targeted.py then attacks.

Reads  : data/preds_<name>.jsonl (the per-row file evaluate.py wrote)
Writes : data/error_analysis.json  (weak intents + confusable pairs + a `targets`
         list [{intent, confuser, n_missed}] that targeted.py consumes)

Run from the track_a_banking77 folder, after Phase 4 has produced the preds file:
    ../../../.venv/bin/python phase5_iterate/error_analysis.py
"""

import argparse
import collections
import json
import os
import sys

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def per_intent(rows, labels):
    """precision / recall / F1 / support per label, from preds rows. pred may be
    None when the model produced nothing we could map to a real label."""
    gold = [r["gold"] for r in rows]
    pred = [r.get("pred") for r in rows]
    out = {}
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        sup = sum(1 for g in gold if g == l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = {"support": sup, "precision": prec, "recall": rec, "f1": f1,
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def confusion(rows):
    """For each gold intent, a Counter of the wrong labels it was predicted as
    (None = the model abstained, tracked separately since it has no confuser)."""
    by = collections.defaultdict(collections.Counter)
    abstain = collections.Counter()
    for r in rows:
        if r.get("correct"):
            continue
        g, p = r["gold"], r.get("pred")
        if p is None:
            abstain[g] += 1
        else:
            by[g][p] += 1
    return by, abstain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="seed-synth",
                    help="condition whose preds_<name>.jsonl to analyze")
    ap.add_argument("--top-intents", type=int, default=12,
                    help="how many weakest intents to target")
    ap.add_argument("--min-f1", type=float, default=0.80,
                    help="only target intents below this F1 (the bar for 'weak')")
    args = ap.parse_args()

    preds_path = f"{common.DATA}/preds_{args.preds}.jsonl"
    if not os.path.exists(preds_path):
        raise SystemExit(f"no preds file at {preds_path}; run "
                         f"eval/evaluate.py --name {args.preds} --adapter <path> first.")

    labels = common.load_labels()
    rows = common.read_jsonl(preds_path)
    m = per_intent(rows, labels)
    by, abstain = confusion(rows)

    # Weakest intents first. A label is a candidate if it is below the bar and the
    # model actually misses it (support > 0). The top confuser is the single wrong
    # label it most often collapses into; that is the pair targeted data will fix.
    ranked = sorted(labels, key=lambda l: (m[l]["f1"], -m[l]["fn"]))
    weak = [l for l in ranked if m[l]["support"] and m[l]["f1"] < args.min_f1][:args.top_intents]

    targets, weak_report = [], []
    for l in weak:
        top = by[l].most_common(1)
        confuser, n = (top[0] if top else (None, 0))
        weak_report.append({
            "intent": l, "f1": m[l]["f1"], "recall": m[l]["recall"],
            "precision": m[l]["precision"], "support": m[l]["support"],
            "top_confuser": confuser, "n_to_confuser": n,
            "n_abstain": abstain.get(l, 0),
        })
        if confuser is not None:
            targets.append({"intent": l, "confuser": confuser, "n_missed": n})

    # Symmetric confusable pairs, ranked by total misses in both directions. This
    # is the same information as a confusion matrix, reduced to the pairs that
    # actually hurt, so the report leads with the real failure modes.
    seen, pairs = set(), []
    for a in labels:
        for b, n in by[a].items():
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            a_to_b = by[a].get(b, 0)
            b_to_a = by[b].get(a, 0)
            pairs.append({"a": key[0], "b": key[1], "a_to_b": by[key[0]].get(key[1], 0),
                          "b_to_a": by[key[1]].get(key[0], 0),
                          "total": by[key[0]].get(key[1], 0) + by[key[1]].get(key[0], 0)})
    pairs.sort(key=lambda d: d["total"], reverse=True)
    top_pairs = pairs[:12]

    out = {
        "preds": preds_path,
        "min_f1": args.min_f1,
        "weakest_intents": weak_report,
        "confusable_pairs": top_pairs,
        "targets": targets,
        "abstain_total": sum(abstain.values()),
    }
    path = f"{common.DATA}/error_analysis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Readable report: the pattern, named.
    print(f"error analysis of {preds_path}")
    print(f"weakest intents (F1 < {args.min_f1}), the targets for iteration:")
    print(f"  {'intent':<34} {'F1':>5} {'rec':>5} {'prec':>5}  most-confused-with")
    for w in weak_report:
        conf = f"{w['top_confuser']} (x{w['n_to_confuser']})" if w["top_confuser"] else "(abstains)"
        print(f"  {w['intent']:<34} {w['f1']:.3f} {w['recall']:.3f} {w['precision']:.3f}  {conf}")
    print(f"\ntop confusable pairs (misses both directions):")
    for p in top_pairs[:8]:
        print(f"  {p['total']:>3}  {p['a']}  <->  {p['b']}  "
              f"({p['a']}->{p['b']} {p['a_to_b']}, {p['b']}->{p['a']} {p['b_to_a']})")
    print(f"\n{len(targets)} targets written -> {path}. Next: targeted.py")


if __name__ == "__main__":
    main()
