"""Phase 4: line up the conditions and read the deltas off both axes.

Reads the per-model result_<name>.json files evaluate.py wrote, lines them up against
a baseline, and prints the before/after: accuracy + macro-F1 on the gold task, and the
three regression probe scores (sentinel / reasoning / tools). The question is whether
the synthetic reasoning data improved resolution prediction without dropping general
reasoning or tool-calling ability.

Run from the track root after the per-model runs exist:
    ../../.venv/bin/python eval/evaluate.py --name base
    ../../.venv/bin/python eval/evaluate.py --name seed       --adapter phase3_train/lora-seed
    ../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
    ../../.venv/bin/python eval/compare.py

Writes data/comparison.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common_c as common


def load_result(name):
    path = f"{common.DATA}/result_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def signed(x):
    return f"{x:+.3f}"


def probe(res, key):
    r = res.get(key)
    return r["score"] if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["base", "seed", "seed-synth"])
    ap.add_argument("--baseline", default="base")
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
            print(f"  ../../.venv/bin/python {cmd.get(name, f'eval/evaluate.py --name {name} --adapter <path>')}")
        print()
    if args.baseline not in results:
        raise SystemExit(f"baseline '{args.baseline}' has no result file; nothing to compare.")
    present = [n for n in args.names if n in results]
    if len(present) < 2:
        raise SystemExit("need the baseline plus at least one other condition to compare.")

    bt = results[args.baseline]["task"]
    print("Phase 4: help-desk resolution, before/after on both axes")
    print(f"base model: {results[args.baseline]['base_model']}")
    print(f"baseline  : {args.baseline}\n")

    print("axis 1 (task)     acc     macroF1  validlbl   dacc    dF1")
    for name in present:
        t = results[name]["task"]
        d_acc = t["accuracy"] - bt["accuracy"]
        d_f1 = t["macro_f1"] - bt["macro_f1"]
        delta = "" if name == args.baseline else f"   {signed(d_acc)}  {signed(d_f1)}"
        print(f"  {name:<14} {t['accuracy']:.3f}   {t['macro_f1']:.3f}    "
              f"{t['valid_label_rate']:.3f}{delta}")

    print("\naxis 2 (regression probes: higher is better)")
    print("  condition      sentinel  reasoning  tools")
    base_p = {k: probe(results[args.baseline], k) for k in ("sentinel", "reasoning", "tools")}
    for name in present:
        cells = []
        for k in ("sentinel", "reasoning", "tools"):
            v = probe(results[name], k)
            if v is None:
                cells.append("  n/a ")
                continue
            d = "" if name == args.baseline or base_p[k] is None else f"({signed(v-base_p[k])})"
            cells.append(f"{v:.3f}{d}")
        print(f"  {name:<14} " + "  ".join(cells))

    summary = {
        "base_model": results[args.baseline]["base_model"],
        "baseline": args.baseline,
        "conditions": {
            name: {
                "adapter": results[name].get("adapter"),
                "task": {k: results[name]["task"][k] for k in
                         ("n", "accuracy", "macro_f1", "valid_label_rate")},
                "sentinel": probe(results[name], "sentinel"),
                "reasoning": probe(results[name], "reasoning"),
                "tools": probe(results[name], "tools"),
            } for name in present
        },
        "missing": missing,
    }
    path = f"{common.DATA}/comparison.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
