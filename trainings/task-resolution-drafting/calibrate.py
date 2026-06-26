"""Calibrate the LLM judge against your own human judgments before trusting it.

The rule: know the judge tracks a person before letting it gate. You hand-judge a handful of
(ticket, reply A, reply B) triples, then this runs the SAME judge on them and reports agreement.
If agreement is high, the win-rate the gate uses is meaningful; if not, fix the judge prompt or
pick a stronger judge before relying on it.

1. Build the calibration set. Easiest: after one pipeline_gen run, the base and candidate answers
   are in runs/result_*.json. Pair them per ticket and hand-label which is better. Or write your
   own data/calibration.jsonl directly, one object per line:
     {"ticket": "...", "answer_a": "...", "answer_b": "...", "human": "a" | "b" | "tie"}
2. Run:  ../../.venv/bin/python calibrate.py
   It prints raw agreement and agreement ignoring ties (the cases where you had a clear preference).

`human` labels which of the two SHOWN answers (a/b) you prefer; the judge sees the same a/b order,
so agreement is a direct person-vs-judge comparison on identical inputs.
"""

import argparse
import json
import os

import common_p2 as c
import judge as J

CALIB = "data/calibration.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=CALIB)
    args = ap.parse_args()
    cfg = c.load_config()
    path = c.resolve(args.file)
    if not os.path.exists(path):
        raise SystemExit(f"no calibration file at {path}; create it (see this file's header)")
    rows = c.read_jsonl(path)
    if not rows:
        raise SystemExit("calibration file is empty")

    jm = cfg["judge"]["model"]
    print(f"[calibrate] {len(rows)} human-judged pairs vs judge {jm}")
    model, tok = J._load_judge(jm)
    try:
        prompts = [J._prompt(r["ticket"], r["answer_a"], r["answer_b"]) for r in rows]
        verdicts = [J._verdict(x) for x in J._gen_verdicts(model, tok, prompts,
                                                           batch=cfg["judge"].get("batch", 4))]
    finally:
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # judge verdict 'A'/'B'/'tie' -> 'a'/'b'/'tie' to compare with human label
    jmap = {"A": "a", "B": "b", "tie": "tie"}
    agree = sum(1 for r, v in zip(rows, verdicts) if r["human"] == jmap[v])
    decisive = [(r, v) for r, v in zip(rows, verdicts) if r["human"] in ("a", "b") and jmap[v] in ("a", "b")]
    agree_dec = sum(1 for r, v in decisive if r["human"] == jmap[v])

    print(f"[calibrate] raw agreement: {agree}/{len(rows)} = {agree/len(rows):.2f}")
    if decisive:
        print(f"[calibrate] agreement on decisive pairs (no ties either side): "
              f"{agree_dec}/{len(decisive)} = {agree_dec/len(decisive):.2f}")
    print("[calibrate] rule of thumb: >= ~0.8 decisive agreement before trusting the judge to gate.")
    for r, v in zip(rows, verdicts):
        flag = "" if r["human"] == jmap[v] else "  <-- disagree"
        print(f"   human={r['human']:<4} judge={jmap[v]:<4}{flag}")


if __name__ == "__main__":
    main()
