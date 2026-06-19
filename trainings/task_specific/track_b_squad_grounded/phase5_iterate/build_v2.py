"""Phase 5, step 3: gate the targeted data and assemble train_synth_v2.

Targeted data gets the SAME discipline as Phase 2, no shortcut for being on
purpose: judge each candidate for faithfulness (reuse the Phase 2 judge),
decontaminate against the sacred gold set, dedup against everything already in
train_synth.jsonl, RAFT-assemble, and append. The v1 set already carries the real
seeds, so the mode-collapse guard rides along.

Reads  : data/gen_targeted.jsonl, data/train_synth.jsonl, data/gold.jsonl,
         data/passages.jsonl
Writes : data/train_synth_v2.jsonl  (retrain with --name seed-synth-v2)

Run from the track_b_squad_grounded folder (Ollama up, for the judge):
    ../../../.venv/bin/python phase5_iterate/build_v2.py
"""

import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2_synthetic"))
import common
import judge

TARGETED = f"{common.DATA}/gen_targeted.jsonl"
BASE_SET = f"{common.DATA}/train_synth.jsonl"
OUT_FILE = f"{common.DATA}/train_synth_v2.jsonl"

SEED = 0
N_DISTRACTORS = 2
GOLD_SIM = 0.70
DUP_SIM = 0.85


def near(shingles, pool, threshold):
    return any(common.jaccard(shingles, s) >= threshold for s in pool)


def main():
    judge.sdg.preflight()
    rng = random.Random(SEED)
    base = common.read_jsonl(BASE_SET)
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    cand = common.read_jsonl(TARGETED)
    pool = common.read_jsonl(f"{common.DATA}/passages.jsonl")

    gold_q_norm = {common.normalize(common.question_of(r)) for r in gold}
    gold_q_sh = [common.char_shingles(common.question_of(r)) for r in gold]
    gold_oracles = {r.get("oracle") for r in gold}

    seen_norm = {common.normalize(common.question_of(r)) for r in base} | gold_q_norm
    kept_q_sh = [common.char_shingles(common.question_of(r)) for r in base]

    kept_rows, balance = [], Counter()
    dropped = {"judge": 0, "gold_q": 0, "gold_passage": 0, "dup": 0}

    for i, r in enumerate(cand, 1):
        q, passage = r["question"].strip(), r["passage"]
        # faithfulness gate (same questions as Phase 2)
        if r["answerable"]:
            faithful, score, _ = judge.judge_answerable(passage, q, r["answer"])
            keep = faithful and score >= judge.KEEP_SCORE
        else:
            keep, _, _ = judge.judge_unanswerable(passage, q)
        if not keep:
            dropped["judge"] += 1
            print(f"[{i}/{len(cand)}] kept={len(kept_rows)}".ljust(40), end="\r")
            continue

        norm, sh = common.normalize(q), common.char_shingles(q)
        if passage in gold_oracles:
            dropped["gold_passage"] += 1
        elif norm in gold_q_norm or near(sh, gold_q_sh, GOLD_SIM):
            dropped["gold_q"] += 1
        elif norm in seen_norm or near(sh, kept_q_sh, DUP_SIM):
            dropped["dup"] += 1
        else:
            distractors = [p["context"] for p in rng.sample(pool, min(len(pool), N_DISTRACTORS * 4))
                           if p["title"] != r["title"] and p["context"] != passage][:N_DISTRACTORS]
            passages = [passage] + distractors
            rng.shuffle(passages)
            answer = r["answer"] if r["answerable"] else common.ABSTAIN
            kept_rows.append(common.build_row(passages, q, answer))
            seen_norm.add(norm)
            kept_q_sh.append(sh)
            balance["answerable" if r["answerable"] else "unanswerable"] += 1
        print(f"[{i}/{len(cand)}] kept={len(kept_rows)}".ljust(40), end="\r")

    final = base + kept_rows
    common.write_jsonl(OUT_FILE, final)

    print()
    print(f"targeted in    : {len(cand)}")
    print(f"kept targeted  : {len(kept_rows)}  "
          f"(answerable={balance['answerable']}, unanswerable={balance['unanswerable']})")
    print(f"dropped        : {dropped}")
    print(f"final v2 rows  : {len(final)}  (= {len(base)} v1 + {len(kept_rows)} targeted)")
    print(f"\nWrote {OUT_FILE}. Next: retrain and re-measure")
    print(f"  ../../../.venv/bin/python phase3_train/train.py "
          f"--data data/train_synth_v2.jsonl --name seed-synth-v2")
    print(f"  ../../../.venv/bin/python eval/evaluate.py --name seed-synth-v2 "
          f"--adapter phase3_train/lora-seed-synth-v2")
    print(f"  ../../../.venv/bin/python eval/compare.py "
          f"--names base seed seed-synth seed-synth-v2 --effect-pair seed-synth seed-synth-v2")


if __name__ == "__main__":
    main()
