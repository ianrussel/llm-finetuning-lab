"""Acceptance gate for the open-ended (generative) task.

Same two-axis idea as the classification gate, but the task axis is the LLM-judge WIN-RATE of the
fine-tuned model over the base model (>= min_win_rate), not a macro-F1 gain. The regression axis is
unchanged: no sentinel / reasoning / tool probe may drop more than max_regression_drop vs base.

  accept iff  judge win-rate >= acceptance.min_win_rate
         AND  no regression probe dropped by more than acceptance.max_regression_drop
"""

PROBES = ["sentinel", "reasoning", "tools"]


def decide(base, cand, win_rate, acceptance):
    min_wr = acceptance["min_win_rate"]
    max_drop = acceptance["max_regression_drop"]

    task_ok = win_rate >= min_wr

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
        f"judge win-rate {win_rate:.3f} vs min {min_wr:.3f} -> {'PASS' if task_ok else 'FAIL'}",
        f"worst regression drop {worst_drop:.3f} vs max {max_drop:.3f} -> {'PASS' if reg_ok else 'FAIL'}",
    ]
    return {
        "accept": accept,
        "win_rate": round(win_rate, 4),
        "worst_regression_drop": round(worst_drop, 4),
        "regressions": regressions,
        "fail_axis": fail_axis,
        "reasons": reasons,
    }
