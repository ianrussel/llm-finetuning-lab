"""Phase 2, quality gate 2: dedup, decontaminate, RAFT-assemble, and balance.

Takes the judge's kept candidates and turns them into the final training set:

  1. keep only judged rows with keep=True.
  2. decontaminate against the SACRED gold set: drop any candidate whose question
     matches a gold question (exact or char-shingle near-dup), or whose passage is
     a gold oracle passage. This protects the evaluation from leaking into training.
  3. dedup: drop candidates that repeat a seed question or an already-kept question
     (exact-normalized or near-duplicate), so no question dominates.
  4. RAFT-assemble: each kept (passage, question, answer) becomes a row whose
     context is the oracle passage plus distractor passages from the pool, shuffled,
     matching how the seed and gold rows are built.
  5. balance report: answerable vs unanswerable counts, so abstention does not get
     swamped, and assemble train_synth.jsonl = ALL seeds + kept synthetic (the real
     seeds stay in as the guard against mode collapse).

Reads  : data/judged.jsonl, data/seed.jsonl, data/gold.jsonl, data/passages.jsonl
Writes : data/train_synth.jsonl  (messages format, ready for Phase 3)

Run from the track_b_squad_grounded folder:
    ../../../.venv/bin/python phase2_synthetic/filter.py
"""

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

JUDGED = f"{common.DATA}/judged.jsonl"
OUT_FILE = f"{common.DATA}/train_synth.jsonl"

SEED = 0
N_DISTRACTORS = 2     # match Phase 1's RAFT width
GOLD_SIM = 0.70       # question near-dup vs a gold question -> contamination, drop
DUP_SIM = 0.85        # question near-dup vs something kept -> redundant, drop


def near(shingles, pool, threshold):
    return any(common.jaccard(shingles, s) >= threshold for s in pool)


def main():
    rng = random.Random(SEED)
    seeds = common.read_jsonl(f"{common.DATA}/seed.jsonl")
    gold = common.read_jsonl(f"{common.DATA}/gold.jsonl")
    judged = common.read_jsonl(JUDGED)
    pool = common.read_jsonl(f"{common.DATA}/passages.jsonl")

    gold_q_norm = {common.normalize(common.question_of(r)) for r in gold}
    gold_q_sh = [common.char_shingles(common.question_of(r)) for r in gold]
    gold_oracles = {r.get("oracle") for r in gold}

    seen_norm = {common.normalize(common.question_of(r)) for r in seeds} | gold_q_norm
    kept_q_sh = [common.char_shingles(common.question_of(r)) for r in seeds]

    kept_rows, balance = [], Counter()
    dropped = {"judge": 0, "gold_q_exact": 0, "gold_q_near": 0, "gold_passage": 0,
               "dup_exact": 0, "dup_near": 0}

    for r in judged:
        if not r.get("keep"):
            dropped["judge"] += 1
            continue
        q = r["question"].strip()
        passage = r["passage"]
        norm = common.normalize(q)
        sh = common.char_shingles(q)
        if passage in gold_oracles:
            dropped["gold_passage"] += 1
            continue
        if norm in gold_q_norm:
            dropped["gold_q_exact"] += 1
            continue
        if near(sh, gold_q_sh, GOLD_SIM):
            dropped["gold_q_near"] += 1
            continue
        if norm in seen_norm:
            dropped["dup_exact"] += 1
            continue
        if near(sh, kept_q_sh, DUP_SIM):
            dropped["dup_near"] += 1
            continue

        # RAFT-assemble: oracle passage + distractors from other articles.
        distractors = [p["context"] for p in rng.sample(pool, min(len(pool), N_DISTRACTORS * 4))
                       if p["title"] != r["title"] and p["context"] != passage][:N_DISTRACTORS]
        passages = [passage] + distractors
        rng.shuffle(passages)
        answer = r["answer"] if r["answerable"] else common.ABSTAIN
        kept_rows.append(common.build_row(passages, q, answer))

        seen_norm.add(norm)
        kept_q_sh.append(sh)
        balance["answerable" if r["answerable"] else "unanswerable"] += 1

    final = list(seeds) + kept_rows
    common.write_jsonl(OUT_FILE, final)

    print(f"judged in        : {len(judged)}")
    print(f"kept synthetic   : {len(kept_rows)}  "
          f"(answerable={balance['answerable']}, unanswerable={balance['unanswerable']})")
    print(f"dropped          : {dropped}")
    print(f"final train rows : {len(final)}  (= {len(seeds)} seeds + {len(kept_rows)} synthetic)")
    print(f"\nWrote {OUT_FILE}. Next: Phase 3 trains the LoRA on this file.")


if __name__ == "__main__":
    main()
