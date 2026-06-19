"""Phase 2, generation (the brief's step 2): per ticket, generate a reasoning trace that
justifies the VERIFIED outcome from the serialized linked context.

The label is real (from the data), so only the reasoning is synthetic, and the
grounding rule holds: the generated answer must equal the given label or the
candidate is dropped. Trace length is varied on purpose (short/medium/long bands)
so the trained model does not learn one fixed reasoning shape.

Reads  : data/train_ids.json, data/field_survey.json (must be confirmed)
Writes : data/gen_reasoning.jsonl  (one candidate per line: id, label, context,
         reasoning, trace_len_bucket)

Run from the track root (Ollama up):
    ../../.venv/bin/python phase2_synthetic/gen_reasoning.py
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import common_c as common
import link
import serialize
import sdg

OUT = f"{common.DATA}/gen_reasoning.jsonl"

GEN_SYSTEM = ("You explain why a help-desk ticket reached a given resolution outcome, "
              "reasoning ONLY from the structured fields and workflow history provided. "
              "The outcome is given and is correct. Do not invent facts that are not in "
              "the context. Reply with ONLY a JSON object.")

# (bucket, instruction, num_predict) - cycled per item to vary trace length.
BANDS = [
    ("short", "in 1 to 2 short sentences", 110),
    ("medium", "in 3 to 4 concise steps", 240),
    ("long", "in 5 to 7 detailed steps", 430),
]


def generate(context, label, band):
    name, instr, npred = band
    raw = sdg.chat([
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content":
            f"{context}\n\nThe verified outcome is: {label}.\n"
            f"Write the reasoning that leads from the fields to this outcome, {instr}, "
            f"then state the outcome. Reply JSON: "
            f'{{"reasoning": "...", "answer": "{label}"}}'},
    ], temperature=0.7, num_predict=npred)
    obj = sdg.parse_json(raw) or {}
    reasoning = str(obj.get("reasoning", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    # strip any stray think tags the model may add; we wrap the trace ourselves
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    return reasoning, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train ids")
    args = ap.parse_args()

    survey = serialize.require_confirmed_survey()
    sdg.preflight()
    db = link.HelpDeskDB()
    train_ids = json.load(open(f"{common.DATA}/train_ids.json"))
    if args.limit:
        train_ids = train_ids[:args.limit]

    written, dropped = 0, 0
    with open(OUT, "w") as f:
        for i, nid in enumerate(train_ids):
            iss = db.issues.get(str(nid))
            if not iss:
                continue
            label = (iss.get("issue_resolution") or "").strip()
            if label not in common.LABELS:
                continue
            context = serialize.serialize_issue(db.get_issue(nid), survey)
            band = BANDS[i % len(BANDS)]
            reasoning, answer = generate(context, label, band)
            # grounding guard: the generated answer must match the verified label
            if not reasoning or common.normalize(answer) != common.normalize(label):
                dropped += 1
            else:
                f.write(json.dumps({"id": str(nid), "label": label, "context": context,
                                    "reasoning": reasoning, "trace_len_bucket": band[0]},
                                   ensure_ascii=False) + "\n")
                written += 1
            print(f"[{i+1}/{len(train_ids)}] kept={written} dropped={dropped}".ljust(50), end="\r")
    print(f"\nWrote {written} candidates ({dropped} dropped) to {OUT}. Next: judge.py")


if __name__ == "__main__":
    main()
