"""Phase 4: line up the conditions and read the deltas off both axes.

evaluate.py scores ONE model and writes data/result_<name>.json. This is the
other half of Phase 4: it reads several of those result files, lines them up
against a baseline (the recorded base run), and prints the before/after the
whole track is built to measure:

  axis 1 (task)       : accuracy, macro-F1, valid-label rate on the gold set
  axis 2 (regression) : sentinel score, to catch forgetting after training

The headline question is "what did the synthetic data add?", so on top of the
table it isolates the seed -> seed-synth move and, from the per-row prediction
files (data/preds_<name>.jsonl), shows which intents gained or regressed.

Run from the track_a_banking77 folder, after the per-model runs exist:

    ../../../.venv/bin/python eval/evaluate.py --name base
    ../../../.venv/bin/python eval/evaluate.py --name seed       --adapter phase3_train/lora-seed
    ../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
    ../../../.venv/bin/python eval/compare.py

Writes data/comparison.json (the table plus the per-intent movement) so the
result survives the session.
"""

import argparse
import json
import os
import sys

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def load_result(name):
    """Read data/result_<name>.json, or None if that run has not been done yet."""
    path = f"{common.DATA}/result_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def per_label_f1(rows, labels):
    """Per-intent F1 from a preds file's rows (each has gold + pred). pred may be
    None when the model produced nothing we could map to a real label."""
    gold = [r["gold"] for r in rows]
    pred = [r.get("pred") for r in rows]
    out = {}
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[l] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def load_preds(name):
    """Per-row predictions for a condition, or None if the file is absent."""
    path = f"{common.DATA}/preds_{name}.jsonl"
    if not os.path.exists(path):
        return None
    return common.read_jsonl(path)


def signed(x):
    """A delta with an explicit sign and a fixed width, so columns line up."""
    return f"{x:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["base", "seed", "seed-synth"],
                    help="conditions to line up, in display order")
    ap.add_argument("--baseline", default="base",
                    help="condition every delta is measured against")
    ap.add_argument("--effect-pair", nargs=2, metavar=("FROM", "TO"),
                    default=["seed", "seed-synth"],
                    help="the before/after pair to break down per intent "
                         "(default isolates what the synthetic data added)")
    args = ap.parse_args()

    # Load what exists; tell the user exactly how to make what does not.
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
            how = cmd.get(name, f"eval/evaluate.py --name {name} --adapter <path>")
            print(f"  ../../../.venv/bin/python {how}")
        print()
    if args.baseline not in results:
        raise SystemExit(f"baseline '{args.baseline}' has no result file; nothing to "
                         f"compare against.")
    present = [n for n in args.names if n in results]
    if len(present) < 2:
        raise SystemExit("need the baseline plus at least one other condition to compare.")

    base = results[args.baseline]
    bt, bs = base["task"], base["sentinel"]

    # axis 1: the task table, with deltas vs the baseline.
    print(f"Phase 4: before/after on both axes")
    print(f"base model: {base['base_model']}")
    print(f"baseline  : {args.baseline}\n")

    print("axis 1 (task)        n      acc     macroF1  validlbl   "
          "dacc    dF1     dvalid")
    for name in present:
        t = results[name]["task"]
        d_acc = t["accuracy"] - bt["accuracy"]
        d_f1 = t["macro_f1"] - bt["macro_f1"]
        d_valid = t["valid_label_rate"] - bt["valid_label_rate"]
        deltas = "" if name == args.baseline else (
            f"   {signed(d_acc)}  {signed(d_f1)}  {signed(d_valid)}")
        print(f"  {name:<16} {t['n']:>5}  {t['accuracy']:.3f}   {t['macro_f1']:.3f}    "
              f"{t['valid_label_rate']:.3f}{deltas}")

    # axis 2: the regression guardrail. A drop here is forgetting, and matters
    # even if axis 1 went up.
    print("\naxis 2 (regression sentinel)")
    for name in present:
        s = results[name]["sentinel"]
        d = s["score"] - bs["score"]
        flag = ""
        if name != args.baseline:
            flag = f"   {signed(d)}" + ("  <- dropped" if d < 0 else "")
        print(f"  {name:<16} {int(s['score'] * s['n'])}/{s['n']}  {s['score']:.3f}{flag}")

    # Isolate one before/after move per intent. By default this is the track's
    # core question (seed -> seed-synth); Phase 5 points it at the iteration move.
    pair = tuple(args.effect_pair)
    intent_moves = None
    if pair[0] in results and pair[1] in results:
        a, b = results[pair[0]]["task"], results[pair[1]]["task"]
        print(f"\nwhat the synthetic data added ({pair[0]} -> {pair[1]}):")
        print(f"  accuracy  {signed(b['accuracy'] - a['accuracy'])}")
        print(f"  macro-F1  {signed(b['macro_f1'] - a['macro_f1'])}")

        # Per-intent movement needs the full per-row preds, not the summary.
        pa, pb = load_preds(pair[0]), load_preds(pair[1])
        if pa and pb:
            labels = common.load_labels()
            fa, fb = per_label_f1(pa, labels), per_label_f1(pb, labels)
            moves = sorted(((fb[l] - fa[l], l) for l in labels), reverse=True)
            gains = [(d, l) for d, l in moves if d > 0][:8]
            regs = [(d, l) for d, l in reversed(moves) if d < 0][:8]
            intent_moves = {"gains": [{"intent": l, "delta_f1": d} for d, l in gains],
                            "regressions": [{"intent": l, "delta_f1": d} for d, l in regs]}
            if gains:
                print("  biggest per-intent F1 gains:")
                for d, l in gains:
                    print(f"     {signed(d)}  {l}")
            if regs:
                print("  per-intent F1 regressions:")
                for d, l in regs:
                    print(f"     {signed(d)}  {l}")
        else:
            print("  (per-intent movement skipped: preds files not found for both runs)")

    # Persist the lined-up numbers so the comparison outlives the session.
    summary = {
        "base_model": base["base_model"],
        "baseline": args.baseline,
        "conditions": {
            name: {
                "adapter": results[name].get("adapter"),
                "task": {k: results[name]["task"][k]
                         for k in ("n", "accuracy", "macro_f1", "valid_label_rate")},
                "sentinel": {k: results[name]["sentinel"][k] for k in ("n", "score")},
            }
            for name in present
        },
        "effect_pair": {"from": pair[0], "to": pair[1]},
        "effect_per_intent": intent_moves,
        "missing": missing,
    }
    path = f"{common.DATA}/comparison.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
