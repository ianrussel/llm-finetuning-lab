"""
Module 4, step 2: fine-tune the embedding model on the mined triplets.

Loss: MultipleNegativesRankingLoss. For each (query, positive, hard-negative) it
pulls the query embedding toward its positive and pushes it away from the hard
negative, and it also uses the other positives in the batch as extra negatives. This
is the standard contrastive setup for retrieval models.

The base model is tiny (all-MiniLM-L6-v2, 22M params), so this trains fast on CPU or
any GPU. Output goes to ./embed-out.

Run from this folder:
    aipy train.py
"""

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from common import BASE_MODEL, OUT_DIR, read_jsonl

EPOCHS = 4
BATCH  = 16

triplets = read_jsonl("data/train_triplets.jsonl")

# One training example per (query, positive, hard-negative) pair.
examples = []
for t in triplets:
    for neg in t["negatives"]:
        examples.append(InputExample(texts=[t["query"], t["positive"], neg]))
print(f"Built {len(examples)} training triplets from {len(triplets)} queries")

model = SentenceTransformer(BASE_MODEL)
loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
loss = losses.MultipleNegativesRankingLoss(model)
warmup = max(1, int(len(loader) * EPOCHS * 0.1))

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup,
    show_progress_bar=True,
)

model.save(OUT_DIR)
print(f"Done. Fine-tuned model saved to {OUT_DIR}. Next: aipy evaluate.py")
