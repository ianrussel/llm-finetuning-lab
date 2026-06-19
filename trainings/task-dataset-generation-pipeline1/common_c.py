"""Shared contract + helpers for the Help Desk pipeline (Track C).

The task: given a help-desk ticket's LINKED structured context (issue metadata, the
workflow time-in-state aggregates, and the change-history timeline), predict the
ticket's resolution outcome and show the reasoning that leads there. This mirrors
the spine of track_a/track_b but the source is a relational knowledge base, not a
single labelled corpus, and the assistant output is a reasoning trace then a label.

Contract
  input  : a serialized linked-context block for one issue (see serialize.py)
  output : "<think>\\n...reason over the fields...\\n</think>\\n\\nThe answer is <LABEL>."
           LABEL drawn from the closed set in data/labels.txt (Done | Won't Do)
  metric : accuracy + macro-F1 on the held-out gold set (macro because the real
           class balance is heavily skewed toward Done)

Layout: this file sits at the pipeline-folder root next to the raw CSVs. RAW is the
folder itself (read-only inputs); DATA is the data/ subfolder for processed artifacts.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = HERE                      # raw CSVs (issues.csv, ...) live alongside this file
DATA = os.path.join(HERE, "data")
LABELS_PATH = os.path.join(DATA, "labels.txt")

# The two-class resolution target. Kept verbatim because the model must reproduce
# the apostrophe/casing exactly.
LABELS = ["Done", "Won't Do"]


def load_labels(path=LABELS_PATH):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def system_prompt(labels):
    """Instruction in front of every example. The model classifies into a closed
    set and must think first, then commit to one label."""
    joined = " or ".join(labels)
    return (
        "You are a help-desk analytics model. You are given one support ticket's "
        "structured context: its metadata, how long it spent in each workflow state, "
        "and its change history. Decide the ticket's final resolution outcome. "
        "First reason step by step inside <think> and </think>, using ONLY the "
        "provided fields. Then, after </think>, reply with exactly one outcome label "
        f"on the last line, copied verbatim. Allowed outcomes: {joined}."
    )


def user_content(context):
    return f"{context}\n\nQuestion: What was this ticket's resolution outcome?"


def assistant_content(reasoning, label):
    return f"<think>\n{reasoning.strip()}\n</think>\n\nThe answer is {label}."


def build_row(context, reasoning, label, labels):
    """A training row: full messages with a reasoning trace + label."""
    return {"messages": [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": user_content(context)},
        {"role": "assistant", "content": assistant_content(reasoning, label)},
    ]}


def build_eval_row(context, label, issue_id, labels):
    """A gold/eval row: prompt (system+user) plus the metadata the scorer needs.
    No assistant turn, since that is what the model must produce."""
    return {"messages": [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": user_content(context)},
    ], "label": label, "id": issue_id}


def user_of(row):
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(s):
    """Lowercase + collapse whitespace, for dedup and label matching."""
    return " ".join(s.lower().split())


def c_predict_label(output, labels):
    """Map a reasoning-then-answer output to one of the closed-set labels, or None.

    The model emits a <think> block then the label, so we score only the tail after
    the last </think> (falling back to the whole output if no tag). Lenient in a way
    that cannot inflate the score: exact match on the last non-empty line first, then
    a verbatim substring search (longest label wins, so 'Won't Do' beats 'Done')."""
    tail = output.rsplit("</think>", 1)[-1] if "</think>" in output else output
    by_norm = {normalize(l): l for l in labels}
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    last = normalize(lines[-1]) if lines else ""
    if last in by_norm:
        return by_norm[last]
    low = normalize(tail)
    hits = [l for l in labels if normalize(l) in low]
    if hits:
        return max(hits, key=lambda l: len(l))
    return None


def char_shingles(s, n=13):
    """Set of overlapping n-char slices of the normalized string, for near-dup and
    contamination checks."""
    t = normalize(s)
    if len(t) <= n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def macro_f1(gold, pred, labels):
    """Unweighted mean F1 across the labels. pred entries may be None."""
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
