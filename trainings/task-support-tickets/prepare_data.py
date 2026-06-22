"""Build a balanced ticket-classification dataset from Tobi-Bueck/customer-support-tickets.

This is the only task-specific code in this folder; everything else (training, eval, gate,
early stopping) is the byte-identical pipeline2 code. It proves the pipeline takes a NEW corpus
and a NEW model with no per-model changes: prepare the data into the same `messages` shape, point
config.yaml at it, run pipeline.py.

The task: given a ticket (subject + body), predict its queue (the support department). Queue is a
closed multi-class label set, so the existing macro-F1 two-axis gate applies unchanged. We filter
to English, balance per label, hold out a sacred gold set, and decontaminate gold vs train.

Run from this folder (CPU is fine, it only downloads + processes):
    ../../.venv/bin/python prepare_data.py
Then review data/labels.txt + data/gold.jsonl, set confirmed: true in config.yaml, run pipeline.py.

Dataset note: schema is detected defensively (LABEL_FIELD with fallbacks). If the column names
differ on the version you pull, adjust LABEL_FIELD / TEXT_FIELDS below; the script prints the
columns it sees so the first run tells you what to set.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REHEARSAL_POOL = os.path.join(HERE, "..", "task-dataset-generation-pipeline1", "data", "train_synth.jsonl")

DATASET = "Tobi-Bueck/customer-support-tickets"
LABEL_FIELD_CANDIDATES = ["queue", "type", "priority", "category"]
TEXT_FIELDS = ["subject", "body"]          # concatenated into the user message
LANG_FIELD = "language"                    # filtered to English if present
KEEP_LANG = "en"

SEED = 0
GOLD_PER_LABEL = 20                        # sacred held-out per queue
SEED_PER_LABEL = 40                        # training examples per queue (LIMA-style small)
MIN_LABEL_COUNT = GOLD_PER_LABEL + SEED_PER_LABEL   # drop queues too rare to fill both
REPLAY_FRACTION = 0.25                     # train_mix = task + ~25% general rehearsal

INSTRUCTION = (
    "You are a support-ticket router. Read the ticket and reply with exactly one queue label "
    "from the list below, copied verbatim, with nothing else.\nQueues:\n{labels}"
)


def norm(s):
    return " ".join(str(s).lower().split())


def pick_label_field(cols):
    for f in LABEL_FIELD_CANDIDATES:
        if f in cols:
            return f
    raise SystemExit(f"no label field found in {cols}; set LABEL_FIELD_CANDIDATES to one of them")


def ticket_text(row):
    parts = [str(row.get(f, "")).strip() for f in TEXT_FIELDS if row.get(f)]
    return "\n".join(p for p in parts if p)


def to_messages(text, labels, label=None):
    msgs = [{"role": "system", "content": INSTRUCTION.format(labels="\n".join(labels))},
            {"role": "user", "content": text}]
    if label is not None:
        msgs.append({"role": "assistant", "content": label})
    return msgs


def main():
    random.seed(SEED)
    os.makedirs(DATA, exist_ok=True)
    print(f"[prep] loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    cols = ds.column_names
    print(f"[prep] columns: {cols}")
    label_field = pick_label_field(cols)
    print(f"[prep] label field: {label_field}")

    # Collect (text, label) for English rows with non-empty text + label.
    rows = []
    for r in ds:
        if LANG_FIELD in cols and norm(r.get(LANG_FIELD)) not in ("", KEEP_LANG, "english"):
            continue
        text, label = ticket_text(r), r.get(label_field)
        if text and label and str(label).strip():
            rows.append((text, str(label).strip()))
    print(f"[prep] usable rows: {len(rows)}")

    # Keep only labels with enough examples to fill gold + seed, then balance.
    counts = Counter(l for _, l in rows)
    labels = sorted(l for l, c in counts.items() if c >= MIN_LABEL_COUNT)
    if not labels:
        raise SystemExit(f"no label has >= {MIN_LABEL_COUNT} rows; counts={dict(counts)}")
    print(f"[prep] kept labels ({len(labels)}): {labels}")

    by_label = defaultdict(list)
    for text, label in rows:
        if label in labels:
            by_label[label].append(text)

    gold, seed_rows, seen = [], [], set()
    for label in labels:
        pool = by_label[label][:]
        random.shuffle(pool)
        # dedup within label by normalized text
        uniq, local = [], set()
        for t in pool:
            k = norm(t)
            if k not in local:
                local.add(k); uniq.append(t)
        g = uniq[:GOLD_PER_LABEL]
        s = uniq[GOLD_PER_LABEL:GOLD_PER_LABEL + SEED_PER_LABEL]
        for t in g:
            gold.append((t, label)); seen.add(norm(t))
        for t in s:
            seed_rows.append((t, label))

    # Decontaminate: drop any seed row whose text also appears in gold.
    before = len(seed_rows)
    seed_rows = [(t, l) for t, l in seed_rows if norm(t) not in seen]
    if before != len(seed_rows):
        print(f"[prep] dropped {before - len(seed_rows)} seed rows overlapping gold")
    random.shuffle(seed_rows)

    # Write labels + gold (prompt only + label) + task-only training set.
    with open(os.path.join(DATA, "labels.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")
    with open(os.path.join(DATA, "gold.jsonl"), "w") as f:
        for text, label in gold:
            f.write(json.dumps({"messages": to_messages(text, labels), "label": label}) + "\n")
    with open(os.path.join(DATA, "train_synth.jsonl"), "w") as f:
        for text, label in seed_rows:
            f.write(json.dumps({"messages": to_messages(text, labels, label)}) + "\n")

    # train_mix = task + a general rehearsal sample (preserve general ability), reusing Part 1's
    # general data as the rehearsal pool if present. If not, replay is off and mix == task.
    mix = [to_messages(t, labels, l) for t, l in seed_rows]
    rehearsal = []
    if os.path.exists(REHEARSAL_POOL):
        pool = [json.loads(x) for x in open(REHEARSAL_POOL) if x.strip()]
        k = int(len(mix) * REPLAY_FRACTION)
        random.shuffle(pool)
        rehearsal = [p["messages"] for p in pool[:k] if "messages" in p]
        print(f"[prep] replay: added {len(rehearsal)} rehearsal rows from Part 1")
    else:
        print("[prep] replay: no rehearsal pool found; train_mix == train_synth (replay effectively off)")
    mixed = mix + rehearsal
    random.shuffle(mixed)
    with open(os.path.join(DATA, "train_mix.jsonl"), "w") as f:
        for msgs in mixed:
            f.write(json.dumps({"messages": msgs}) + "\n")

    print(f"[prep] wrote: labels={len(labels)}, gold={len(gold)}, "
          f"train_synth={len(seed_rows)}, train_mix={len(mixed)}")
    print("[prep] review data/labels.txt and data/gold.jsonl, then set confirmed: true in config.yaml")


if __name__ == "__main__":
    main()
