"""Phase 2, generation pass 1: paraphrase expansion (the simplest, safest pass).

For each seed query we ask the local model for several reworded versions that
keep the SAME meaning and therefore the SAME intent label. This is the grounding
rule in action: vary the phrasing freely, never change the verified label. No new
facts, just variety in wording, length, and tone.

Reads  : data/seed.jsonl   (the 385 verified seeds from Phase 1)
Writes : data/gen_paraphrase.jsonl  (one {user, intent, method, seed} per line)

Run from the track_a_banking77 folder (plain `python` is the wrong interpreter):
    ../../../.venv/bin/python phase2_synthetic/paraphrase.py
"""

import json
import os
import sys

# common.py lives one level up; sdg.py sits next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # parent folder -> common.py
sys.path.insert(0, _HERE)                    # this folder -> sdg.py
import common
import sdg

OUT_FILE = f"{common.DATA}/gen_paraphrase.jsonl"
N_PER_SEED = 5

GEN_SYSTEM = ("You rewrite a bank customer's message. You keep the original meaning, "
              "topic, and request exactly the same, so the support category never "
              "changes. You never invent new facts. You never add greetings, "
              "sign-offs, or signatures. You output only the rewritten message text.")

PREAMBLE = ("here are", "sure", "certainly", "rewrite", "rewrites", "message:",
            "here's", "below are", "versions:")


def paraphrase(message, k):
    text = sdg.chat([
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content":
            f"Rewrite this bank customer message in {k} different ways. Keep the same "
            f"meaning and the same request. Vary the wording, length, and tone (some "
            f"terse, some polite, some with a typo or slang). Each line must be one "
            f"complete standalone version. Do NOT include greetings, sign-offs, "
            f"numbering, or any extra text.\n\nMessage: \"{message}\""},
    ])
    out = []
    for raw in text.splitlines():
        s = sdg.clean_line(raw)
        if len(s.split()) < 3 or s.lower().startswith(PREAMBLE):
            continue
        out.append(s)
        if len(out) >= k:
            break
    return out


def main():
    sdg.preflight()
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    written = 0
    with open(OUT_FILE, "w") as f:
        for i, row in enumerate(seeds, 1):
            message = common.user_of(row)
            intent = common.assistant_of(row)
            for v in paraphrase(message, N_PER_SEED):
                f.write(json.dumps({"user": v, "intent": intent,
                                    "method": "paraphrase", "seed": message},
                                   ensure_ascii=False) + "\n")
                written += 1
            print(f"[{i}/{len(seeds)}] {intent:<28} {message[:40]!r}".ljust(90), end="\r")
    print(f"\nWrote {written} paraphrases to {OUT_FILE}. Next: evolve.py")


if __name__ == "__main__":
    main()
