"""
Module 4, step 1: hard-negative mining.

A "negative" is a document that is NOT the right answer for a query. A *hard*
negative is one the current model wrongly ranks high, a near-miss that looks similar
but is wrong. Training against hard negatives is what teaches the model to pull the
right document above its confusing neighbours. Random negatives teach almost nothing
because they are already easy to separate.

How we mine them: embed the corpus with the base model, and for each training query
take the top-ranked documents that are not the labelled positive. Those are exactly
the model's current confusions. Output triplets (query, positive, negatives) to
data/train_triplets.jsonl for train.py.

Run from this folder:
    aipy mine_negatives.py
"""

import json

from sentence_transformers import SentenceTransformer, util

from common import BASE_MODEL, read_jsonl, load_corpus

N_NEG = 3            # hard negatives per query
OUT   = "data/train_triplets.jsonl"

doc_ids, doc_texts, id2text = load_corpus()
train = read_jsonl("data/train.jsonl")

print(f"Embedding {len(doc_texts)} documents with the base model...")
model = SentenceTransformer(BASE_MODEL)
doc_emb = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

written = 0
with open(OUT, "w") as f:
    for row in train:
        query, pos_id = row["query"], row["positive_id"]
        q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, doc_emb)[0]
        ranked = sims.argsort(descending=True).tolist()

        negatives = []
        for idx in ranked:                       # top-ranked first
            if doc_ids[idx] == pos_id:
                continue                          # skip the correct answer
            negatives.append(doc_texts[idx])
            if len(negatives) >= N_NEG:
                break

        f.write(json.dumps({"query": query,
                            "positive": id2text[pos_id],
                            "negatives": negatives}) + "\n")
        written += 1

print(f"Wrote {written} triplets ({N_NEG} hard negatives each) to {OUT}. Next: aipy train.py")
