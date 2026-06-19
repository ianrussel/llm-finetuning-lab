"""Phase 5, step 3: gate the targeted data and assemble train_synth_v2.

Targeted data gets the SAME discipline as Phase 2, with no shortcut just because
it was generated on purpose. In order:

  1. judge each candidate for faithfulness (reuse the Phase 2 LLM-as-judge), keep
     only faithful rows scoring >= KEEP_SCORE.
  2. decontaminate against the SACRED gold set (exact + char-shingle near-dup).
     This is the non-negotiable step; targeted generation must never leak gold.
  3. dedup against everything already in train_synth.jsonl and against each other,
     so the new rows are genuinely new and no phrasing dominates.
  4. assemble train_synth_v2.jsonl = train_synth.jsonl + kept targeted rows.

The v1 set already keeps the real seeds in the mix, so carrying it forward keeps
that mode-collapse guard intact.

Reads  : data/gen_targeted.jsonl, data/train_synth.jsonl, data/gold.jsonl
Writes : data/train_synth_v2.jsonl  (retrain with --name seed-synth-v2)

Run from the track_a_banking77 folder (Ollama up, for the judge):
    ../../../.venv/bin/python phase5_iterate/build_v2.py
"""

import os
import sys
from collections import Counter

# common.py one level up; judge.py lives in phase2_synthetic (reuse its verdict).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2_synthetic"))
import common
import judge

TARGETED = f"{common.DATA}/gen_targeted.jsonl"
BASE_SET = f"{common.DATA}/train_synth.jsonl"   # the v1 set we extend
OUT_FILE = f"{common.DATA}/train_synth_v2.jsonl"

GOLD_SIM = 0.70   # near-dup vs a gold query -> contamination, drop
DUP_SIM = 0.85    # near-dup vs something already in the set -> redundant, drop


def near(shingles, pool, threshold):
    return any(common.jaccard(shingles, s) >= threshold for s in pool)


def main():
    judge.sdg.preflight()
    labels = common.load_labels()
    base = common.read_jsonl(BASE_SET)        # all of v1 (seeds + v1 synthetic)
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    cand = common.read_jsonl(TARGETED)

    gold_norm = {common.normalize(common.user_of(r)) for r in gold}
    gold_sh = [common.char_shingles(common.user_of(r)) for r in gold]

    # seen = every message already in v1 (and gold), so targeted rows must be new.
    seen_norm = {common.normalize(common.user_of(r)) for r in base} | gold_norm
    kept_sh = [common.char_shingles(common.user_of(r)) for r in base]

    kept_rows, per_intent = [], Counter()
    dropped = {"judge": 0, "bad_intent": 0, "gold_exact": 0, "gold_near": 0,
               "dup_exact": 0, "dup_near": 0}

    for i, r in enumerate(cand, 1):
        intent, user = r["intent"], r["user"].strip()
        if intent not in labels:
            dropped["bad_intent"] += 1
            continue
        faithful, score, _ = judge.verdict(user, intent, labels)
        if not (faithful and score >= judge.KEEP_SCORE):
            dropped["judge"] += 1
            print(f"[{i}/{len(cand)}] kept={len(kept_rows)}".ljust(40), end="\r")
            continue
        norm = common.normalize(user)
        sh = common.char_shingles(user)
        if norm in gold_norm:
            dropped["gold_exact"] += 1
        elif near(sh, gold_sh, GOLD_SIM):
            dropped["gold_near"] += 1
        elif norm in seen_norm:
            dropped["dup_exact"] += 1
        elif near(sh, kept_sh, DUP_SIM):
            dropped["dup_near"] += 1
        else:
            seen_norm.add(norm)
            kept_sh.append(sh)
            kept_rows.append(common.build_row(user, intent, labels))
            per_intent[intent] += 1
        print(f"[{i}/{len(cand)}] kept={len(kept_rows)}".ljust(40), end="\r")

    final = base + kept_rows
    common.write_jsonl(OUT_FILE, final)

    print()
    print(f"targeted in      : {len(cand)}")
    print(f"kept targeted    : {len(kept_rows)}")
    print(f"dropped          : {dropped}")
    print(f"final v2 rows    : {len(final)}  (= {len(base)} v1 + {len(kept_rows)} targeted)")
    if per_intent:
        print(f"kept per intent  : "
              + ", ".join(f"{l}={per_intent[l]}" for l in sorted(per_intent)))
    print(f"\nWrote {OUT_FILE}. Next: retrain and re-measure")
    print(f"  ../../../.venv/bin/python phase3_train/train.py "
          f"--data data/train_synth_v2.jsonl --name seed-synth-v2")
    print(f"  ../../../.venv/bin/python eval/evaluate.py --name seed-synth-v2 "
          f"--adapter phase3_train/lora-seed-synth-v2")
    print(f"  ../../../.venv/bin/python eval/compare.py "
          f"--names base seed seed-synth seed-synth-v2 --effect-pair seed-synth seed-synth-v2")


if __name__ == "__main__":
    main()
