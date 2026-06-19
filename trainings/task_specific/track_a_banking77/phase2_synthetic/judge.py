"""Phase 2, quality gate 1: LLM-as-judge faithfulness filter.

Generation drifts. A paraphrase can blur into a neighbouring intent, and an
evolved message can quietly become a different request. Before any dedup or
training we ask the local model, acting as a judge, one question per candidate:
does this message really express its assigned intent, and not a different one?

Output per candidate: a faithfulness verdict plus a 1-5 score. We keep only
verdicts that are faithful AND score at or above KEEP_SCORE.

Known biases to stay honest about (the MT-Bench / LLM-as-judge caveat):
  - a judge tends to favour longer, more fluent text (length bias)
  - the generator and the judge are the SAME model here, so it can be lenient on
    its own style (self-preference). A stronger or different judge model is the
    real fix; for a learning run, same-model judging still catches obvious drift.

Reads  : data/gen_paraphrase.jsonl, data/gen_evol.jsonl
Writes : data/judged.jsonl  (every candidate, annotated with the verdict)

Run from the track_a_banking77 folder:
    ../../../.venv/bin/python phase2_synthetic/judge.py
"""

import json
import os
import re
import sys

# common.py lives one level up; sdg.py sits next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # parent folder -> common.py
sys.path.insert(0, _HERE)                    # this folder -> sdg.py
import common
import sdg

INPUTS = [f"{common.DATA}/gen_paraphrase.jsonl", f"{common.DATA}/gen_evol.jsonl"]
OUT_FILE = f"{common.DATA}/judged.jsonl"
KEEP_SCORE = 4   # 1-5; keep faithful candidates scoring this or higher

JUDGE_SYSTEM = ("You are a strict labeller for a bank support intent dataset. You decide "
                "whether a customer message truly expresses one specific intent. You reply "
                "with ONLY a JSON object and nothing else.")


def verdict(message, intent, labels):
    raw = sdg.chat([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content":
            f"Allowed intents: {', '.join(labels)}\n\n"
            f"Message: \"{message}\"\n"
            f"Claimed intent: {intent}\n\n"
            f"Does the message clearly and unambiguously express the claimed intent, "
            f"and not some other intent in the list? Reply with JSON: "
            f'{{"faithful": true or false, "score": 1-5, "reason": "short"}}'},
    ], temperature=0.0, num_predict=120)
    return parse(raw)


def parse(text):
    """Lenient parse: pull the JSON object if present, else read flags out of text."""
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        faithful = bool(obj.get("faithful"))
        score = int(obj.get("score", 0))
        return faithful, score, str(obj.get("reason", ""))[:120]
    except Exception:
        low = text.lower()
        faithful = "true" in low and "false" not in low.split("true")[0][-12:]
        m = re.search(r'([1-5])', text)
        return faithful, (int(m.group(1)) if m else 0), "unparsed"


def main():
    sdg.preflight()
    labels = common.load_labels()
    rows = [r for path in INPUTS for r in common.read_jsonl(path)]
    kept = 0
    with open(OUT_FILE, "w") as f:
        for i, r in enumerate(rows, 1):
            faithful, score, reason = verdict(r["user"], r["intent"], labels)
            keep = faithful and score >= KEEP_SCORE
            kept += keep
            r = {**r, "faithful": faithful, "score": score,
                 "reason": reason, "keep": keep}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] kept={kept}".ljust(40), end="\r")
    print(f"\nJudged {len(rows)} candidates, kept {kept}. Wrote {OUT_FILE}. Next: filter.py")


if __name__ == "__main__":
    main()
