"""Shared bits for module 4 (embedding retrieval)."""

import json

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small (22M), 384-dim
OUT_DIR    = "./embed-out"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path="data/corpus.jsonl"):
    """Return (ids, texts, id->text) for the document collection."""
    docs = read_jsonl(path)
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    return ids, texts, {d["id"]: d["text"] for d in docs}
