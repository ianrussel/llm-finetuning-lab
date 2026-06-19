"""Phase 2, quality gate 1: LLM-as-judge faithfulness filter.

Generation drifts, and for a grounded task the drift is dangerous in a specific
way: an "answerable" pair whose answer is not really supported teaches the model
to hallucinate, and an "unanswerable" question that the passage actually DOES
answer teaches it to abstain when it should not. So the judge asks a different
question per candidate type:

  - answerable   : is the proposed answer correct AND fully supported by the
                   passage? Keep only faithful pairs scoring >= KEEP_SCORE.
  - unanswerable : can this question be answered using only this passage? Keep
                   only the ones the judge confirms are genuinely unanswerable.

Known biases to stay honest about (same as Track A): the generator and the judge
are the SAME local model, so it can be lenient on its own output (self-preference),
and judges favour fluent text (length bias). A stronger, different judge model is
the real fix; same-model judging still catches obvious drift.

Reads  : data/gen_qa.jsonl
Writes : data/judged.jsonl  (every candidate, annotated with the verdict)

Run from the track_b_squad_grounded folder (Ollama up):
    ../../../.venv/bin/python phase2_synthetic/judge.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import common
import sdg

IN_FILE = f"{common.DATA}/gen_qa.jsonl"
OUT_FILE = f"{common.DATA}/judged.jsonl"
KEEP_SCORE = 4   # 1-5; answerable pairs must score at least this to be kept

GRADER = ("You are a strict grader for a reading-comprehension dataset. You judge a "
          "question against ONE passage and reply with ONLY a JSON object, nothing else.")


def judge_answerable(passage, question, answer):
    raw = sdg.chat([
        {"role": "system", "content": GRADER},
        {"role": "user", "content":
            f"Passage:\n{passage}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
            f"Is the proposed answer correct AND fully supported by the passage (found in "
            f"or directly entailed by it, using no outside knowledge)? Reply JSON: "
            f'{{"faithful": true or false, "score": 1-5, "reason": "short"}}'},
    ], temperature=0.0, num_predict=120)
    obj = sdg.parse_json(raw) or {}
    return bool(obj.get("faithful")), int(obj.get("score", 0) or 0), str(obj.get("reason", ""))[:120]


def judge_unanswerable(passage, question):
    raw = sdg.chat([
        {"role": "system", "content": GRADER},
        {"role": "user", "content":
            f"Passage:\n{passage}\n\nQuestion: {question}\n\n"
            f"Can this question be answered using ONLY this passage? Reply JSON: "
            f'{{"answerable": true or false, "reason": "short"}}'},
    ], temperature=0.0, num_predict=100)
    obj = sdg.parse_json(raw) or {}
    # keep it only if the judge agrees it is genuinely unanswerable
    answerable = bool(obj.get("answerable"))
    return (not answerable), (5 if not answerable else 0), str(obj.get("reason", ""))[:120]


def main():
    sdg.preflight()
    rows = common.read_jsonl(IN_FILE)
    kept = 0
    with open(OUT_FILE, "w") as f:
        for i, r in enumerate(rows, 1):
            if r["answerable"]:
                faithful, score, reason = judge_answerable(r["passage"], r["question"], r["answer"])
                keep = faithful and score >= KEEP_SCORE
            else:
                faithful, score, reason = judge_unanswerable(r["passage"], r["question"])
                keep = faithful
            kept += keep
            out = {**r, "faithful": faithful, "score": score, "reason": reason, "keep": keep}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] kept={kept}".ljust(40), end="\r")
    print(f"\nJudged {len(rows)} candidates, kept {kept}. Wrote {OUT_FILE}. Next: filter.py")


if __name__ == "__main__":
    main()
