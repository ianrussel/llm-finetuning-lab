"""Phase 2, quality gate 1: LLM-as-judge faithfulness filter for the reasoning traces.

A generated trace is only useful if it actually follows from the ticket's fields and
validly supports the verified outcome. The judge reads each candidate and asks exactly
that, dropping traces that invent facts not in the context or that do not support the
label. Same honesty caveat as the other tracks: generator and judge are the same local
model (self-preference, length bias); a stronger judge model is the real fix.

Reads  : data/gen_reasoning.jsonl
Writes : data/judged.jsonl  (every candidate annotated with the verdict)

Run from the track root (Ollama up):
    ../../.venv/bin/python phase2_synthetic/judge.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common
import sdg

IN_FILE = f"{common.DATA}/gen_reasoning.jsonl"
OUT_FILE = f"{common.DATA}/judged.jsonl"
KEEP_SCORE = 4   # 1-5; keep faithful traces scoring this or higher

JUDGE_SYSTEM = ("You are a strict reviewer of reasoning for a help-desk outcome "
                "classifier. You check whether a reasoning trace is grounded in the "
                "given fields and validly supports the stated outcome. You reply with "
                "ONLY a JSON object.")


def verdict(context, reasoning, label):
    raw = sdg.chat([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content":
            f"{context}\n\nStated outcome: {label}\n\nReasoning trace:\n{reasoning}\n\n"
            f"Does the reasoning follow ONLY from the fields above and validly support "
            f"the stated outcome, without inventing facts or contradicting the fields? "
            f'Reply JSON: {{"faithful": true or false, "score": 1-5, "reason": "short"}}'},
    ], temperature=0.0, num_predict=120)
    obj = sdg.parse_json(raw) or {}
    return bool(obj.get("faithful")), int(obj.get("score", 0) or 0), str(obj.get("reason", ""))[:120]


def main():
    sdg.preflight()
    rows = common.read_jsonl(IN_FILE)
    kept = 0
    with open(OUT_FILE, "w") as f:
        for i, r in enumerate(rows, 1):
            faithful, score, reason = verdict(r["context"], r["reasoning"], r["label"])
            keep = faithful and score >= KEEP_SCORE
            kept += keep
            f.write(json.dumps({**r, "faithful": faithful, "score": score,
                                "reason": reason, "keep": keep}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] kept={kept}".ljust(40), end="\r")
    print(f"\nJudged {len(rows)} traces, kept {kept}. Wrote {OUT_FILE}. Next: filter.py")


if __name__ == "__main__":
    main()
