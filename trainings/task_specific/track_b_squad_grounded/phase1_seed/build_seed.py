"""Phase 1: define the grounded task and build the seed + sacred gold sets.

Loads SQuAD v2 and writes three files into data/:

  - gold.jsonl   : the sacred held-out eval set, from the VALIDATION split. A
                   balanced mix of answerable and unanswerable questions, each
                   assembled with RAFT distractors, plus the metadata the scorer
                   needs (gold answers, answerable flag, the oracle passage for
                   the faithfulness judge). Nothing derived from this set may ever
                   enter training.
  - seed.jsonl   : a small verified training seed, from the TRAIN split. Real
                   (passage, question, answer) triples in the messages format,
                   answerable + unanswerable, RAFT-assembled. The honest control
                   for "what does the synthetic data add?".
  - passages.jsonl : a pool of unique TRAIN passages for Phase 2 to generate new
                   grounded QA from, and to draw distractors from.

Disciplines baked in (the same ones Track A enforces):
  - gold comes from validation, seed from train, so they are disjoint by split.
  - a normalized-question decontamination check drops any seed row whose question
    also appears in gold, and asserts zero overlap at the end.
  - the run is deterministic (fixed SEED) so the split reproduces.

Run from the track_b_squad_grounded folder:
    ../../../.venv/bin/python phase1_seed/build_seed.py
"""

import os
import random
import sys

from datasets import load_dataset

# common.py lives one level up (shared by every phase); make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

DATASET = "rajpurkar/squad_v2"
SEED = 0
N_DISTRACTORS = 2          # RAFT: passages added around the oracle, per row

SEED_ANSWERABLE = 300
SEED_UNANSWERABLE = 100    # 400 seed rows, 3:1 answerable:unanswerable
GOLD_ANSWERABLE = 400
GOLD_UNANSWERABLE = 200    # 600 gold rows, same ratio
N_PASSAGES = 800           # unique train passages kept for Phase 2 generation


def is_answerable(row):
    return len(row["answers"]["text"]) > 0


def assemble(rng, oracle_ctx, title, question, answers, answerable, distractor_pool):
    """Build one RAFT row: oracle passage + N distractors from other articles,
    shuffled so the answer is not always in the same slot."""
    pool = [c for (t, c) in distractor_pool if t != title and c != oracle_ctx]
    distractors = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
    passages = [oracle_ctx] + distractors
    rng.shuffle(passages)
    row = common.build_gold_row(passages, question, answers, answerable)
    row["oracle"] = oracle_ctx
    return row


def sample_split(split, n_ans, n_unans, rng, distractor_pool):
    ans = [r for r in split if is_answerable(r)]
    unans = [r for r in split if not is_answerable(r)]
    rng.shuffle(ans)
    rng.shuffle(unans)
    picked = ans[:n_ans] + unans[:n_unans]
    rng.shuffle(picked)
    rows = []
    for r in picked:
        rows.append(assemble(rng, r["context"], r["title"], r["question"],
                             r["answers"]["text"], is_answerable(r), distractor_pool))
    return rows


def main():
    rng = random.Random(SEED)
    print(f"loading {DATASET} ...")
    ds = load_dataset(DATASET)
    train, val = ds["train"], ds["validation"]

    # Distractor pool: unique (title, context) pairs from train. Dedup on context.
    seen, pool = set(), []
    for r in train:
        c = r["context"]
        if c not in seen:
            seen.add(c)
            pool.append((r["title"], c))
    print(f"unique train passages: {len(pool)}")

    gold = sample_split(val, GOLD_ANSWERABLE, GOLD_UNANSWERABLE, rng, pool)
    seed = sample_split(train, SEED_ANSWERABLE, SEED_UNANSWERABLE, rng, pool)

    # Decontaminate: drop any seed row whose question matches a gold question.
    gold_q = {common.normalize(common.question_of(r)) for r in gold}
    before = len(seed)
    seed = [r for r in seed if common.normalize(common.question_of(r)) not in gold_q]
    dropped = before - len(seed)

    # Passage pool for Phase 2 generation (exclude passages used as a gold oracle,
    # so generated QA cannot be grounded in a gold passage).
    gold_oracles = {r["oracle"] for r in gold}
    passages = [{"title": t, "context": c} for (t, c) in pool if c not in gold_oracles]
    rng.shuffle(passages)
    passages = passages[:N_PASSAGES]

    os.makedirs(common.DATA, exist_ok=True)
    common.write_jsonl(f"{common.DATA}/gold.jsonl", gold)
    common.write_jsonl(f"{common.DATA}/seed.jsonl", seed)
    common.write_jsonl(f"{common.DATA}/passages.jsonl", passages)

    # Independent checks.
    seed_q = {common.normalize(common.question_of(r)) for r in seed}
    assert seed_q.isdisjoint(gold_q), "seed/gold question overlap after decontamination"
    g_ans = sum(1 for r in gold if r["answerable"])
    s_ans = sum(1 for r in seed if r["answerable"])
    print(f"gold : {len(gold)} rows ({g_ans} answerable, {len(gold) - g_ans} unanswerable)")
    print(f"seed : {len(seed)} rows ({s_ans} answerable, {len(seed) - s_ans} unanswerable), "
          f"decontaminated {dropped}")
    print(f"passages pool for Phase 2: {len(passages)}")
    print(f"RAFT: {N_DISTRACTORS} distractors per row")
    print("wrote data/gold.jsonl, data/seed.jsonl, data/passages.jsonl")


if __name__ == "__main__":
    main()
