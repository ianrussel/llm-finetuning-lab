"""
Module 3: quality-filter the paraphrases and assemble the synthetic training set.

A model that paraphrases will sometimes repeat itself, echo a seed verbatim, drift
off-task, or accidentally reproduce a held-out eval message. Synthetic data is only
useful after a cleanup pass, so this is where the "judge-filter / eyeball" step lives.

Steps:
  - keep all seeds as-is
  - for each paraphrase: drop empties, bad labels, exact duplicates, near-duplicates,
    and anything that collides with the held-out eval set (no leakage)
Writes data/train_synth.jsonl = seeds + kept paraphrases, in the messages format
train.py expects.

Run from this folder:
    aipy filter.py
"""

import json
from difflib import SequenceMatcher

from common import (CATEGORIES, PRIORITIES, build_row, read_jsonl,
                    user_of, normalize, looks_like_junk)

GENERATED   = "data/generated.jsonl"
SEEDS_FILE  = "data/seeds.jsonl"
EVAL_FILE   = "data/eval.jsonl"
OUT_FILE    = "data/train_synth.jsonl"
SIM_THRESHOLD = 0.92        # drop a paraphrase this similar to one already kept

seeds = read_jsonl(SEEDS_FILE)
generated = read_jsonl(GENERATED)
eval_rows = read_jsonl(EVAL_FILE)

eval_users = {normalize(user_of(r)) for r in eval_rows}

kept = list(seeds)                       # always keep the seeds
seen = {normalize(user_of(r)) for r in seeds}


def too_similar(norm_text):
    for s in seen:
        if SequenceMatcher(None, norm_text, s).ratio() >= SIM_THRESHOLD:
            return True
    return False


added = 0
dropped = {"empty": 0, "junk": 0, "bad_label": 0, "eval_leak": 0, "duplicate": 0}
for g in generated:
    user = (g.get("user") or "").strip()
    cat, pri = g.get("category"), g.get("priority")
    if not user:
        dropped["empty"] += 1
        continue
    if looks_like_junk(user):           # greetings, sign-offs, too-short fragments
        dropped["junk"] += 1
        continue
    if cat not in CATEGORIES or pri not in PRIORITIES:
        dropped["bad_label"] += 1
        continue
    norm = normalize(user)
    if norm in eval_users:
        dropped["eval_leak"] += 1
        continue
    if norm in seen or too_similar(norm):
        dropped["duplicate"] += 1
        continue
    seen.add(norm)
    kept.append(build_row(user, cat, pri))
    added += 1

with open(OUT_FILE, "w") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")

print(f"seeds: {len(seeds)}  generated: {len(generated)}  "
      f"kept paraphrases: {added}  total train rows: {len(kept)}")
print("dropped:", dropped)
print(f"Wrote {OUT_FILE}. Next: aipy train.py")
