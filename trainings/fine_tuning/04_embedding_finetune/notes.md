# Module 4 notes: embedding fine-tuning concepts

Quick reference I wrote to familiarize myself with the ideas behind this module. Tied
back to the actual scripts (mine_negatives.py, train.py, evaluate.py).

## Contrastive learning, positives and negatives

- Contrastive learning trains by comparison, not by predicting a label. It teaches the
  model: these two texts should be close in vector space, those two should be far apart.
- For embeddings you have an anchor (here the user query) plus:
  - Positive: a text that should match the anchor. Here, the correct help article for a
    query (query "Cancel my plan please" -> positive doc cancel-subscription).
  - Negative: a text that should not match (a different, wrong article).
- Training pulls the anchor toward its positive and pushes it away from its negatives,
  measured by cosine similarity. Over many pairs the model learns a space where relevant
  query/doc pairs land near each other.
- That is why retrieval works after: embed the query, embed the docs, nearest doc is the
  answer (what evaluate.py does).

## Hard-negative mining, why it matters, false negatives

- Hard-negative mining = picking negatives that are difficult: docs the model currently
  thinks are similar to the query but are actually wrong (near-misses), instead of
  obviously unrelated ones.
- Why it matters: a random negative is already far away, so it gives almost no learning
  signal. A hard negative (pause-subscription vs the correct cancel-subscription) sits
  right next to the positive, so correcting it teaches the fine distinctions that improve
  ranking.
- How this module mines them: mine_negatives.py embeds the corpus with the base model,
  ranks docs per query, and takes the top-ranked docs that are not the labeled positive.
  Those are exactly the model's current confusions.
- Risk of false negatives: some of those top-ranked "negatives" might actually be
  relevant and just unlabeled. Training pushes the query away from them, teaching the
  model something wrong and hurting accuracy.
  - Mitigations: skip candidates whose similarity is too close to the positive (a
    margin/threshold), verify with a strong model or by hand, make sure labels are
    complete, or cap how many hard negatives per query.

## MultipleNegativesRankingLoss and Matryoshka training

- MultipleNegativesRankingLoss (MNRL), the loss in train.py:
  - Takes (anchor, positive) pairs, optionally with a hard negative. Our examples are
    InputExample(texts=[query, positive, hard_negative]).
  - Within a batch, for each anchor it treats its own positive as the one correct answer
    and every other positive in the batch (plus your hard negatives) as negatives, then
    applies softmax/cross-entropy over the cosine similarities.
  - Effect: pull anchor and positive together, push anchor and everything-else apart, all
    at once. Bigger batch = more free in-batch negatives = generally better.
  - It is the standard strong default for retrieval fine-tuning.
- Matryoshka representation training (MRL):
  - Trains the embedding so truncated prefixes of the vector are themselves good
    embeddings, like nested dolls: first 64 dims work, first 128 better, full 384 best.
  - Lets you shorten embeddings on demand (search with 64 or 128 dims instead of 384) to
    save memory and speed up search, with only a small quality drop.
  - In sentence-transformers it is a wrapper (MatryoshkaLoss) around a base loss like
    MNRL, training several truncation sizes at once. This module does not use it, but you
    could wrap MNRL to get a size-flexible model.

## NDCG@k, recall@k, MRR, and before/after

- Recall@k: did a relevant doc show up in the top k? Binary per query, then averaged.
  evaluate.py reports Recall@1 (correct doc is #1) and Recall@3. Ignores exact position
  within the top k.
- MRR (mean reciprocal rank): average of 1 / rank of the first relevant doc. Rank 1 ->
  1.0, rank 2 -> 0.5, rank 5 -> 0.2. Rewards getting the right answer high. Good when
  there is one correct answer per query (our case). We use MRR@10.
- NDCG@k (normalized discounted cumulative gain): the richest metric. Rewards relevant
  docs and their position (a 1/log2(rank+1) discount), supports graded relevance (some
  docs more relevant than others), and is normalized against the ideal ordering so it
  lands in [0,1]. Best with graded labels. Our toy task has one correct doc per query, so
  Recall and MRR are enough.
- How to compute before/after on your own data (the evaluate.py recipe):
  1. Hold out a labeled query set: each query mapped to its correct doc(s).
  2. With the base model: embed the corpus and each query, rank docs by cosine, find
     where the correct doc lands, compute Recall@k / MRR / NDCG@k.
  3. Fine-tune, then repeat the identical evaluation with the fine-tuned model.
  4. Compare the two metric sets (and per-query ranks) side by side.
  - Discipline: eval queries must be held out (not trained on) and the metric must be
    identical for both models, or the comparison is meaningless. In our run the base
    already scored 1.00, so there was no headroom, which is itself a valid finding.
