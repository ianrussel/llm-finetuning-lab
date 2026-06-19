"""Shared bits for Track A: the banking77 intent classifier.

The task contract lives here so the seed builder, the (later) synthetic
generator, the trainer and the evaluator all speak the exact same format.

Contract
  input  : one customer banking query (a short free-text string)
  output : exactly one intent label, drawn from the 77-label closed set,
           emitted verbatim with no extra words, punctuation or quotes
  metric : accuracy and macro-F1 on the held-out gold set
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LABELS_PATH = os.path.join(DATA, "labels.txt")


def load_labels(path=LABELS_PATH):
    """The closed set of 77 intents, in a fixed order (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def system_prompt(labels):
    """The instruction we put in front of every example.

    We list the whole label set so the model is classifying into a known,
    closed vocabulary rather than inventing free text. Output is the bare
    label, which makes scoring a plain string match.
    """
    joined = ", ".join(labels)
    return (
        "You are a banking customer-support intent classifier. "
        "Read the customer message and reply with EXACTLY ONE intent label "
        "from the list below, copied verbatim, with nothing else (no quotes, "
        "no punctuation, no explanation).\n"
        f"Allowed intents: {joined}"
    )


def build_row(query, intent, labels):
    """One example in the conversational messages format the repo uses."""
    return {"messages": [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": query},
        {"role": "assistant", "content": intent},
    ]}


def user_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def assistant_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "assistant")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(s):
    """Lowercase + collapse whitespace, for dedup and decontamination."""
    return " ".join(s.lower().split())


def predict_label(output, labels):
    """Map a free-text model output to one of the closed-set labels, or None.

    A small base model will not always emit a clean label, so we are lenient in
    a way that still cannot inflate the score: first try an exact match on the
    first line, then look for a label that appears verbatim inside the output
    (longest wins, so `top_up_failed` is preferred over a shorter substring).
    None means the model produced nothing we can map to a valid intent.
    """
    by_norm = {normalize(l): l for l in labels}
    first = normalize(output.splitlines()[0]) if output.strip() else ""
    if first in by_norm:
        return by_norm[first]
    low = normalize(output)
    hits = [l for l in labels if normalize(l) in low]
    if hits:
        return max(hits, key=lambda l: len(l))
    return None


def char_shingles(s, n=13):
    """Set of overlapping n-character slices of the normalized string.

    Used for near-duplicate and contamination checks: two messages that share
    most of their ~13-char shingles are effectively the same text even if a word
    was swapped. Short strings collapse to a single shingle.
    """
    t = normalize(s)
    if len(t) <= n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    """Overlap of two shingle sets in [0, 1]."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def macro_f1(gold, pred, labels):
    """Unweighted mean F1 across all labels. pred entries may be None."""
    total = 0.0
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        total += f1
    return total / len(labels)
