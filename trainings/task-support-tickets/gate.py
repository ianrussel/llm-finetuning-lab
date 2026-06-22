"""Stage 3.4 (the acceptance gate): accept the adapter only if it clears BOTH axes.

Pure logic, no torch. Given the base and candidate result dicts (from
evaluate_from_config) and the acceptance thresholds, decide accept/reject and say
which axis failed, so the orchestrator knows whether to do its adjusted re-run.

  accept  iff  task macro-F1 gain >= min_task_gain_macro_f1
          AND  no regression probe dropped by more than max_regression_drop
"""

import argparse
import json
import os

import common_p2 as c

PROBES = ["sentinel", "reasoning", "tools"]


def decide(base, cand, acceptance):
    min_gain = acceptance["min_task_gain_macro_f1"]
    max_drop = acceptance["max_regression_drop"]

    task_gain = cand["task"]["macro_f1"] - base["task"]["macro_f1"]
    task_ok = task_gain >= min_gain

    regressions = {}
    worst_drop = 0.0
    for k in PROBES:
        b, cd = base.get(k), cand.get(k)
        if not b or not cd:
            continue
        delta = cd["score"] - b["score"]
        regressions[k] = round(delta, 4)
        worst_drop = max(worst_drop, -delta)
    reg_ok = worst_drop <= max_drop

    accept = task_ok and reg_ok
    fail_axis = []
    if not task_ok:
        fail_axis.append("task")
    if not reg_ok:
        fail_axis.append("regression")

    reasons = [
        f"task macro-F1 gain {task_gain:+.3f} vs min {min_gain:+.3f} -> {'PASS' if task_ok else 'FAIL'}",
        f"worst regression drop {worst_drop:.3f} vs max {max_drop:.3f} -> {'PASS' if reg_ok else 'FAIL'}",
    ]
    return {
        "accept": accept,
        "task_gain": round(task_gain, 4),
        "worst_regression_drop": round(worst_drop, 4),
        "regressions": regressions,
        "fail_axis": fail_axis,
        "reasons": reasons,
    }


def _load_result(cfg, name):
    path = os.path.join(c.out_dir(cfg), f"result_{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no result file at {path}; run evaluate_from_config.py --name {name}")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    cfg = c.load_config()
    verdict = decide(_load_result(cfg, args.base), _load_result(cfg, args.candidate),
                     cfg["acceptance"])
    print(f"GATE: {'ACCEPT' if verdict['accept'] else 'REJECT'}")
    for r in verdict["reasons"]:
        print("  -", r)
    print("  per-probe deltas:", verdict["regressions"])
    if not verdict["accept"]:
        print("  failed axis:", verdict["fail_axis"])


if __name__ == "__main__":
    main()
