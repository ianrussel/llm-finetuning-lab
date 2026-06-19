# Module 4: Embedding fine-tune

Goal: fine-tune a small embedding model with hard-negative mining, and measure a
retrieval metric before and after on my own small set.

This module is a different flavor from 1 to 3. Those fine-tuned a generative model
(produce text). Here the model produces a vector (an embedding), and the job is
retrieval: given a query, rank a collection of documents so the right one is on top.

## The task

A tiny help-center search. data/corpus.jsonl is 18 short help articles. The queries
ask the same things in different words, so the model has to match on meaning, not
keywords. "My credit card expired, how do I put in a new one?" should retrieve the
"update-card" article even though they share almost no words.

## Key idea: hard-negative mining

A negative is a document that is not the answer for a query. A *hard* negative is one
the model currently ranks high by mistake, a near-miss. Training against hard negatives
is what teaches the model to separate the right doc from its confusing neighbors.
Random negatives barely help because they are already easy to tell apart.

mine_negatives.py finds them automatically: embed the corpus with the base model, and
for each query take the top-ranked docs that are not the correct one. Those are exactly
the current confusions.

## The metric

evaluate.py ranks every doc for each held-out query and reports:
- Recall@1: the correct doc is the top hit
- Recall@3: the correct doc is in the top 3
- MRR@10:   mean reciprocal rank (1 / rank of the correct doc)

It scores the base model and, once ./embed-out exists, the fine-tuned model too, side by
side, plus a per-query rank table.

## Files

- data/corpus.jsonl: 18 help articles (id, text).
- data/train.jsonl: 18 queries, each with the id of its correct article.
- data/eval.jsonl: 8 held-out queries (different phrasings), never trained on.
- data/train_triplets.jsonl: query + positive + hard negatives (made by mine_negatives.py).
- common.py: shared constants and loaders.
- mine_negatives.py, train.py, evaluate.py.

## Setup (once)

This module needs the embeddings library. Install it through python -m pip (the
.venv/bin/pip script has a stale shebang from when the folder was renamed):

```bash
.venv/bin/python -m pip install sentence-transformers
```

The base model (all-MiniLM-L6-v2, 22M params, 384-dim) downloads on first use (~90 MB).
It is tiny, so everything runs fast on CPU or any GPU.

## How to run

From this folder:

```bash
aipy evaluate.py        # before: base model only (no ./embed-out yet)
aipy mine_negatives.py  # -> data/train_triplets.jsonl
aipy train.py           # -> ./embed-out
aipy evaluate.py        # after: base vs fine-tuned, side by side
```

## What to look for

- The base model is already decent (this is a general-purpose embedding model), so the
  before numbers will not be zero. The question is whether fine-tuning on your domain
  pushes the right docs higher, watch Recall@1 and the per-query ranks.
- If fine-tuning does not help, common causes: too few training queries, hard negatives
  that are actually correct answers (label noise), or the base model already saturating
  this easy corpus. Add queries or harder distractor docs.

## Make it my own

- Change N_NEG in mine_negatives.py (more or fewer hard negatives per query).
- Swap BASE_MODEL in common.py (e.g. BAAI/bge-small-en-v1.5) and compare.
- Grow the corpus with more, more-similar articles so retrieval is genuinely harder.
