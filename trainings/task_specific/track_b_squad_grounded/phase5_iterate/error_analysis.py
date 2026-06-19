"""Phase 5, step 1: error-analyze the grounded-QA misses and NAME the failure mode.

A grounded-QA model fails in three distinct ways, and each calls for different
data, so iteration has to name which one is dominant:

  - hallucination   : on an UNANSWERABLE question it gave an answer instead of
                      abstaining. Fix: more (and harder) unanswerable examples.
  - over-abstention : on an ANSWERABLE question it said "not in the context" when
                      the answer was there. Fix: more answerable examples.
  - wrong answer    : on an ANSWERABLE question it answered but with low token-F1.
                      Fix: more answerable comprehension examples.

Reads  : data/preds_<name>.jsonl (the per-row file evaluate.py wrote)
Writes : data/error_analysis.json  (the bucket counts + a `targets` list of modes
         to boost, [{mode, weight, reason}], that targeted.py consumes)

Run from the track_b_squad_grounded folder, after Phase 4:
    ../../../.venv/bin/python phase5_iterate/error_analysis.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

HALLUCINATION_BAR = 0.15   # above this unanswerable-hallucination rate -> target it
ANSWERABLE_BAR = 0.30      # above this answerable-error rate -> target it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="seed-synth",
                    help="condition whose preds_<name>.jsonl to analyze")
    args = ap.parse_args()

    preds_path = f"{common.DATA}/preds_{args.preds}.jsonl"
    if not os.path.exists(preds_path):
        raise SystemExit(f"no preds file at {preds_path}; run "
                         f"eval/evaluate.py --name {args.preds} --adapter <path> first.")
    rows = common.read_jsonl(preds_path)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    # answerable failure split
    wrong = [r for r in ans if not r["correct"] and not r["abstained"]]
    over_abstained = [r for r in ans if r["abstained"]]
    # unanswerable failure
    hallucinated = [r for r in unans if not r["correct"]]

    halluc_rate = len(hallucinated) / len(unans) if unans else 0.0
    ans_err_rate = (len(wrong) + len(over_abstained)) / len(ans) if ans else 0.0
    over_abs_rate = len(over_abstained) / len(ans) if ans else 0.0

    targets = []
    if halluc_rate > HALLUCINATION_BAR:
        targets.append({"mode": "unanswerable", "weight": round(halluc_rate, 3),
                        "reason": f"hallucination rate {halluc_rate:.3f} on unanswerable rows"})
    if ans_err_rate > ANSWERABLE_BAR:
        targets.append({"mode": "answerable", "weight": round(ans_err_rate, 3),
                        "reason": f"answerable error rate {ans_err_rate:.3f} "
                                  f"({len(wrong)} wrong, {len(over_abstained)} over-abstained)"})

    out = {
        "preds": preds_path,
        "answerable": {"n": len(ans), "wrong": len(wrong),
                       "over_abstained": len(over_abstained),
                       "error_rate": ans_err_rate, "over_abstention_rate": over_abs_rate},
        "unanswerable": {"n": len(unans), "hallucinated": len(hallucinated),
                         "hallucination_rate": halluc_rate},
        "targets": targets,
    }
    path = f"{common.DATA}/error_analysis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"error analysis of {preds_path}")
    print(f"  answerable   : {len(ans)} rows, {len(wrong)} wrong, "
          f"{len(over_abstained)} over-abstained (error rate {ans_err_rate:.3f})")
    print(f"  unanswerable : {len(unans)} rows, {len(hallucinated)} hallucinated "
          f"(rate {halluc_rate:.3f})")
    if targets:
        print("  targets for iteration:")
        for t in targets:
            print(f"    boost {t['mode']:<13} ({t['reason']})")
    else:
        print("  no mode over its bar; model is balanced, consider stopping.")
    print(f"\nwrote {path}. Next: targeted.py")


if __name__ == "__main__":
    main()
