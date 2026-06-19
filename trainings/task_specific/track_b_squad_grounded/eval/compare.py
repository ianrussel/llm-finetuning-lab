"""Phase 4: line up the conditions and read the deltas off both axes.

evaluate.py scores ONE grounded-QA model and writes data/result_<name>.json. This
reads several of those, lines them up against a baseline, and prints the before/
after the track is built to measure:

  axis 1 (task)       : grounded_score, plus the answerable (EM/F1) and
                        unanswerable (abstention accuracy, hallucination rate) parts
  axis 2 (regression) : the sentinel score, to catch forgetting after training

The track's question is "what did the document-grounded synthetic data add?", so
on top of the table it isolates one before/after pair (default seed -> seed-synth)
and shows the move on each sub-metric, which is where grounded QA usually shifts:
better abstention and less hallucination from the synthetic unanswerable data.

Run from the track_b_squad_grounded folder, after the per-model runs exist:

    ../../../.venv/bin/python eval/evaluate.py --name base
    ../../../.venv/bin/python eval/evaluate.py --name seed       --adapter phase3_train/lora-seed
    ../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
    ../../../.venv/bin/python eval/compare.py

Writes data/comparison.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def load_result(name):
    path = f"{common.DATA}/result_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def signed(x):
    return f"{x:+.3f}"


# metrics where higher is better, and the one (hallucination) where lower is better
HIGHER = ["grounded_score", "em", "f1", "abstention_accuracy"]
LOWER = ["hallucination_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["base", "seed", "seed-synth"])
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--effect-pair", nargs=2, metavar=("FROM", "TO"),
                    default=["seed", "seed-synth"])
    args = ap.parse_args()

    cmd = {
        "base": "eval/evaluate.py --name base",
        "seed": "eval/evaluate.py --name seed       --adapter phase3_train/lora-seed",
        "seed-synth": "eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth",
    }
    results, missing = {}, []
    for name in args.names:
        r = load_result(name)
        if r is None:
            missing.append(name)
        else:
            results[name] = r
    if missing:
        print("missing result files (run these first):")
        for name in missing:
            print(f"  ../../../.venv/bin/python {cmd.get(name, f'eval/evaluate.py --name {name} --adapter <path>')}")
        print()
    if args.baseline not in results:
        raise SystemExit(f"baseline '{args.baseline}' has no result file; nothing to compare.")
    present = [n for n in args.names if n in results]
    if len(present) < 2:
        raise SystemExit("need the baseline plus at least one other condition to compare.")

    bt = results[args.baseline]["task"]
    bs = results[args.baseline]["sentinel"]

    print("Phase 4: grounded-QA before/after on both axes")
    print(f"base model: {results[args.baseline]['base_model']}")
    print(f"baseline  : {args.baseline}\n")

    print("axis 1 (task)      grounded   EM      F1     abst_acc  halluc   "
          "dGround")
    for name in present:
        t = results[name]["task"]
        dg = t["grounded_score"] - bt["grounded_score"]
        delta = "" if name == args.baseline else f"   {signed(dg)}"
        print(f"  {name:<14} {t['grounded_score']:.3f}    {t['em']:.3f}  {t['f1']:.3f}  "
              f"{t['abstention_accuracy']:.3f}    {t['hallucination_rate']:.3f}{delta}")

    print("\naxis 2 (regression sentinel)")
    for name in present:
        s = results[name]["sentinel"]
        d = s["score"] - bs["score"]
        flag = "" if name == args.baseline else f"   {signed(d)}" + ("  <- dropped" if d < 0 else "")
        print(f"  {name:<14} {int(s['score']*s['n'])}/{s['n']}  {s['score']:.3f}{flag}")

    # Isolate one before/after move across every sub-metric.
    pair = tuple(args.effect_pair)
    effect = None
    if pair[0] in results and pair[1] in results:
        a, b = results[pair[0]]["task"], results[pair[1]]["task"]
        print(f"\nwhat the move {pair[0]} -> {pair[1]} changed:")
        effect = {}
        for k in HIGHER + LOWER:
            d = b[k] - a[k]
            effect[k] = d
            arrow = "better" if ((k in HIGHER and d > 0) or (k in LOWER and d < 0)) else \
                    ("worse" if d != 0 else "flat")
            print(f"  {k:<20} {a[k]:.3f} -> {b[k]:.3f}   ({signed(d)}, {arrow})")

    summary = {
        "base_model": results[args.baseline]["base_model"],
        "baseline": args.baseline,
        "conditions": {
            name: {
                "adapter": results[name].get("adapter"),
                "task": {k: results[name]["task"][k] for k in
                         ("n", "grounded_score", "em", "f1", "abstention_accuracy",
                          "hallucination_rate")},
                "sentinel": {k: results[name]["sentinel"][k] for k in ("n", "score")},
            }
            for name in present
        },
        "effect_pair": {"from": pair[0], "to": pair[1]},
        "effect": effect,
        "missing": missing,
    }
    path = f"{common.DATA}/comparison.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
