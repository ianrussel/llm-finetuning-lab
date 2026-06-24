"""Build a balanced issue-type classification dataset from the public Jira issues dataset
(arXiv:2201.08368, Montgomery et al., "An Alternative Issue Tracking Dataset of Public Jira
Repositories"). Task: given an issue's summary + description, predict its type (Bug / Improvement /
New Feature / Task / ...). Closed label set, so task1's macro-F1 two-axis gate applies unchanged.

This is the only task-specific file in task2; the six *.py modules are byte-identical to task1.

DATA SOURCE (important): unlike Tobi-Bueck, this dataset is not a one-line HuggingFace load. It ships
as large per-repository dumps (Zenodo, MongoDB/JSON). Get it once, then point SOURCE at it:
  - SOURCE = a local .jsonl where each line is one issue (flat or with a nested "fields" object), OR
  - SOURCE = a HuggingFace dataset id, if you use a mirror.
The script detects the type/summary/description fields defensively (flat or under "fields") and
prints the columns + a sample on first run so you can confirm the schema. Adjust the *_PATHS below
if your export names things differently.

Run from this folder (CPU is fine):
    ../../../.venv/bin/python prepare_data.py
Then review data/labels.txt + data/gold.jsonl, set confirmed: true in config.yaml, run pipeline.py.
"""

import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REHEARSAL_POOL = os.path.join(HERE, "..", "..", "task-dataset-generation-pipeline1", "data", "train_synth.jsonl")

SOURCE = os.path.join(DATA, "jira_issues.jsonl")   # local JSONL you create (see header), or a HF dataset id
# field paths tried in order; each is a list of keys walked into the (possibly nested) issue object
LABEL_PATHS = [["type"], ["issuetype"], ["fields", "issuetype", "name"], ["fields", "type", "name"]]
SUMMARY_PATHS = [["summary"], ["fields", "summary"], ["title"]]
DESC_PATHS = [["description"], ["fields", "description"]]

SEED = 0
GOLD_PER_LABEL = 20
SEED_PER_LABEL = 40
MIN_LABEL_COUNT = GOLD_PER_LABEL + SEED_PER_LABEL
DESC_MAX_CHARS = 1500          # Jira descriptions can be huge; cap so rows fit max_seq_len
REPLAY_FRACTION = 0.25

INSTRUCTION = ("You are an issue triager. Read the issue and reply with exactly one issue-type "
               "label from the list below, copied verbatim, with nothing else.\nTypes:\n{labels}")


def norm(s):
    return " ".join(str(s).lower().split())


def dig(obj, path):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def first(obj, paths):
    for p in paths:
        v = dig(obj, p)
        if v not in (None, ""):
            return v
    return None


def load_rows():
    if SOURCE.endswith(".jsonl") and os.path.exists(SOURCE):
        return [json.loads(l) for l in open(SOURCE) if l.strip()]
    if os.path.exists(SOURCE):  # plain .json array
        return json.load(open(SOURCE))
    # otherwise treat SOURCE as a HF dataset id
    from datasets import load_dataset
    return list(load_dataset(SOURCE, split="train"))


def issue_text(r):
    summary = first(r, SUMMARY_PATHS) or ""
    desc = (first(r, DESC_PATHS) or "")[:DESC_MAX_CHARS]
    return "\n".join(x for x in [str(summary).strip(), str(desc).strip()] if x)


def to_messages(text, labels, label=None):
    m = [{"role": "system", "content": INSTRUCTION.format(labels="\n".join(labels))},
         {"role": "user", "content": text}]
    if label is not None:
        m.append({"role": "assistant", "content": label})
    return m


def main():
    random.seed(SEED)
    os.makedirs(DATA, exist_ok=True)
    print(f"[prep] loading Jira issues from: {SOURCE}")
    rows_raw = load_rows()
    if not rows_raw:
        raise SystemExit(f"no rows from {SOURCE}; see this file's header for how to get the dataset")
    print(f"[prep] sample keys: {sorted(list(rows_raw[0].keys()))[:12]}")

    rows = []
    for r in rows_raw:
        text, label = issue_text(r), first(r, LABEL_PATHS)
        if text and label and str(label).strip():
            rows.append((text, str(label).strip()))
    print(f"[prep] usable rows: {len(rows)}")
    if not rows:
        raise SystemExit("could not extract (text, type); adjust LABEL_PATHS/SUMMARY_PATHS to your schema")

    counts = Counter(l for _, l in rows)
    labels = sorted(l for l, c in counts.items() if c >= MIN_LABEL_COUNT)
    if not labels:
        raise SystemExit(f"no type has >= {MIN_LABEL_COUNT} rows; counts={dict(counts)}")
    print(f"[prep] kept types ({len(labels)}): {labels}")

    by_label = defaultdict(list)
    for text, label in rows:
        if label in labels:
            by_label[label].append(text)

    gold, seed_rows, seen = [], [], set()
    for label in labels:
        pool = by_label[label][:]
        random.shuffle(pool)
        uniq, local = [], set()
        for t in pool:
            k = norm(t)
            if k not in local:
                local.add(k); uniq.append(t)
        for t in uniq[:GOLD_PER_LABEL]:
            gold.append((t, label)); seen.add(norm(t))
        for t in uniq[GOLD_PER_LABEL:GOLD_PER_LABEL + SEED_PER_LABEL]:
            seed_rows.append((t, label))
    before = len(seed_rows)
    seed_rows = [(t, l) for t, l in seed_rows if norm(t) not in seen]
    if before != len(seed_rows):
        print(f"[prep] dropped {before - len(seed_rows)} seed rows overlapping gold")
    random.shuffle(seed_rows)

    with open(os.path.join(DATA, "labels.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")
    with open(os.path.join(DATA, "gold.jsonl"), "w") as f:
        for text, label in gold:
            f.write(json.dumps({"messages": to_messages(text, labels), "label": label}) + "\n")
    with open(os.path.join(DATA, "train_synth.jsonl"), "w") as f:
        for text, label in seed_rows:
            f.write(json.dumps({"messages": to_messages(text, labels, label)}) + "\n")

    mix = [to_messages(t, labels, l) for t, l in seed_rows]
    reh = []
    if os.path.exists(REHEARSAL_POOL):
        pool = [json.loads(x) for x in open(REHEARSAL_POOL) if x.strip()]
        random.shuffle(pool)
        reh = [p["messages"] for p in pool[:int(len(mix) * REPLAY_FRACTION)] if "messages" in p]
        print(f"[prep] replay: added {len(reh)} rehearsal rows from Part 1")
    else:
        print("[prep] replay: no rehearsal pool found; train_mix == train_synth")
    mixed = mix + reh
    random.shuffle(mixed)
    with open(os.path.join(DATA, "train_mix.jsonl"), "w") as f:
        for m in mixed:
            f.write(json.dumps({"messages": m}) + "\n")

    print(f"[prep] wrote labels={len(labels)}, gold={len(gold)}, "
          f"train_synth={len(seed_rows)}, train_mix={len(mixed)}")
    print("[prep] review data/labels.txt and data/gold.jsonl, then set confirmed: true in config.yaml")


if __name__ == "__main__":
    main()
