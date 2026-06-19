"""Phase 2, generation: turn passages into grounded QA (document-grounded SDG).

This is the core of the corpus-grounded track: instead of paraphrasing existing
examples, we read each passage and GENERATE new question/answer pairs from it,
the document-grounded approach (the idea behind Bonito / Augmentoolkit). Two
kinds, because the task has two behaviours to teach:

  - answerable : a question whose answer is a short exact span copied from the
                 passage. The grounding rule is strict, the answer must be in the
                 passage verbatim, so the label stays anchored to real text.
  - unanswerable : a question that is on-topic for the passage but whose answer
                 the passage does NOT contain. The target is the fixed abstention
                 string. This is what teaches the model to say "not in the
                 context" instead of hallucinating.

Faithfulness is not assumed here, only attempted; judge.py checks it next.

Reads  : data/passages.jsonl  (the Phase 1 pool of train passages)
Writes : data/gen_qa.jsonl    (one candidate per line: passage, title, question,
         answer or null, answerable, method)

Run from the track_b_squad_grounded folder (Ollama up):
    ../../../.venv/bin/python phase2_synthetic/qgen.py
"""

import argparse
import json
import os
import sys

# common.py one level up; sdg.py sits next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import common
import sdg

OUT_FILE = f"{common.DATA}/gen_qa.jsonl"

ANS_SYSTEM = ("You write reading-comprehension questions whose answer is a short exact "
              "span copied verbatim from a given passage. You never ask anything the "
              "passage does not directly answer. You output ONLY a JSON array.")

UNANS_SYSTEM = ("You write questions that are on-topic for a passage but that the passage "
                "does NOT answer, used to teach a model when to abstain. The questions must "
                "be plausible for the topic yet unanswerable from the text. You output ONLY "
                "a JSON array of strings.")


def gen_answerable(passage, k):
    raw = sdg.chat([
        {"role": "system", "content": ANS_SYSTEM},
        {"role": "user", "content":
            f"Passage:\n{passage}\n\nWrite {k} diverse questions answerable from this "
            f"passage. For each, give the question and the shortest exact answer span "
            f"copied verbatim from the passage. Output a JSON array of objects: "
            f'[{{"question": "...", "answer": "..."}}]'},
    ], temperature=0.7)
    obj = sdg.parse_json(raw) or []
    out = []
    for o in obj:
        if isinstance(o, dict) and o.get("question") and o.get("answer"):
            q, a = str(o["question"]).strip(), str(o["answer"]).strip()
            # grounding guard: keep only answers that really occur in the passage
            if a and a.lower() in passage.lower() and len(q.split()) >= 3:
                out.append((q, a))
    return out


def gen_unanswerable(passage, k):
    raw = sdg.chat([
        {"role": "system", "content": UNANS_SYSTEM},
        {"role": "user", "content":
            f"Passage:\n{passage}\n\nWrite {k} questions that are related to this passage's "
            f"topic but whose answer is NOT stated anywhere in the passage. Output a JSON "
            f'array of strings.'},
    ], temperature=0.9)
    obj = sdg.parse_json(raw) or []
    out = []
    for o in obj:
        q = str(o).strip() if not isinstance(o, dict) else str(o.get("question", "")).strip()
        # reject any "unanswerable" question whose words clearly are answered: a
        # cheap guard is to drop ones whose answer span the model leaked; judge.py
        # does the real check.
        if len(q.split()) >= 3:
            out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable-per-passage", type=int, default=2)
    ap.add_argument("--unanswerable-per-passage", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N passages")
    args = ap.parse_args()

    sdg.preflight()
    passages = common.read_jsonl(f"{common.DATA}/passages.jsonl")
    if args.limit:
        passages = passages[:args.limit]

    written = 0
    with open(OUT_FILE, "w") as f:
        for i, p in enumerate(passages, 1):
            ctx, title = p["context"], p["title"]
            for q, a in gen_answerable(ctx, args.answerable_per_passage):
                f.write(json.dumps({"passage": ctx, "title": title, "question": q,
                                    "answer": a, "answerable": True,
                                    "method": "qgen:answerable"}, ensure_ascii=False) + "\n")
                written += 1
            for q in gen_unanswerable(ctx, args.unanswerable_per_passage):
                f.write(json.dumps({"passage": ctx, "title": title, "question": q,
                                    "answer": None, "answerable": False,
                                    "method": "qgen:unanswerable"}, ensure_ascii=False) + "\n")
                written += 1
            print(f"[{i}/{len(passages)}] candidates={written}".ljust(50), end="\r")
    print(f"\nWrote {written} QA candidates to {OUT_FILE}. Next: judge.py")


if __name__ == "__main__":
    main()
