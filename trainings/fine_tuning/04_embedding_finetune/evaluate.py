"""
Module 4, step 3: measure retrieval quality before and after.

For each held-out query we rank every document by cosine similarity and check where
the correct one lands. Metrics:
  - Recall@1: correct doc is the top result (fraction of queries)
  - Recall@3: correct doc is in the top 3
  - MRR@10:   mean reciprocal rank (1/rank of the correct doc, 0 if past rank 10)

It scores the base model, and if ./embed-out exists it scores the fine-tuned model too
and prints both so the before/after is side by side.

Run from this folder:
    aipy evaluate.py
"""

import os

from sentence_transformers import SentenceTransformer, util

from common import BASE_MODEL, OUT_DIR, read_jsonl, load_corpus

doc_ids, doc_texts, _ = load_corpus()
eval_rows = read_jsonl("data/eval.jsonl")


def evaluate(model):
    doc_emb = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)
    r1 = r3 = mrr = 0
    per_query = []
    for row in eval_rows:
        q_emb = model.encode(row["query"], convert_to_tensor=True,
                             normalize_embeddings=True)
        sims = util.cos_sim(q_emb, doc_emb)[0]
        ranked = [doc_ids[i] for i in sims.argsort(descending=True).tolist()]
        rank = ranked.index(row["positive_id"]) + 1
        if rank == 1:
            r1 += 1
        if rank <= 3:
            r3 += 1
        if rank <= 10:
            mrr += 1.0 / rank
        per_query.append((row["query"], row["positive_id"], rank))
    n = len(eval_rows)
    return (r1 / n, r3 / n, mrr / n), per_query


print("Scoring base model...")
base = SentenceTransformer(BASE_MODEL)
(b1, b3, bmrr), base_pq = evaluate(base)

ft_scores = None
if os.path.isdir(OUT_DIR):
    print("Scoring fine-tuned model...")
    ft = SentenceTransformer(OUT_DIR)
    (f1, f3, fmrr), ft_pq = evaluate(ft)
    ft_scores = (f1, f3, fmrr)

print()
print(f"{'':12} {'Recall@1':>9} {'Recall@3':>9} {'MRR@10':>8}")
print(f"{'base':12} {b1:9.2f} {b3:9.2f} {bmrr:8.2f}")
if ft_scores:
    print(f"{'fine-tuned':12} {ft_scores[0]:9.2f} {ft_scores[1]:9.2f} {ft_scores[2]:8.2f}")

    print("\nPer-query rank of the correct doc (lower is better):")
    print(f"{'query':45} {'base':>5} {'ft':>5}")
    for (q, _pid, brank), (_q2, _p2, frank) in zip(base_pq, ft_pq):
        print(f"{q[:45]:45} {brank:>5} {frank:>5}")
else:
    print("\n(no ./embed-out yet, run train.py to get the after numbers)")
