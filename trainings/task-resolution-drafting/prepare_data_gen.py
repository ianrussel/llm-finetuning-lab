"""Build the open-ended reply-drafting dataset from Tobi-Bueck/customer-support-tickets.

Same corpus as the queue-classification task, but the target is the agent's prose reply (the
`answer` field), not a label. So the task is open-ended: draft a resolution for the ticket. The
gold set keeps the reference answer for calibration/reference; the gate scores quality with the
LLM judge (judge.py), not against the reference.

Run from this folder (CPU is fine):
    ../../.venv/bin/python prepare_data_gen.py
Then review data/gold.jsonl, set confirmed: true in config.yaml, run pipeline_gen.py.
"""

import json
import os
import random

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REHEARSAL_POOL = os.path.join(HERE, "..", "task-dataset-generation-pipeline1", "data", "train_synth.jsonl")

DATASET = "Tobi-Bueck/customer-support-tickets"
TEXT_FIELDS = ["subject", "body"]
ANSWER_FIELD_CANDIDATES = ["answer", "response", "agent_answer", "reply"]
LANG_FIELD, KEEP_LANG = "language", "en"

SEED = 0
N_GOLD = 60            # held-out tickets the judge scores base-vs-candidate on
N_TRAIN = 400          # training examples (ticket -> reference reply)
MIN_ANSWER_CHARS = 40  # drop trivially short answers
REPLAY_FRACTION = 0.25

INSTRUCTION = ("You are a customer-support agent. Read the ticket and write a single helpful, "
               "professional reply that resolves it. Reply with the message only.")


def norm(s):
    return " ".join(str(s).lower().split())


def ticket_text(r):
    return "\n".join(str(r.get(f, "")).strip() for f in TEXT_FIELDS if r.get(f))


def messages(ticket, answer=None):
    m = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": ticket}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m


def main():
    random.seed(SEED)
    os.makedirs(DATA, exist_ok=True)
    print(f"[prep-gen] loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    cols = ds.column_names
    print(f"[prep-gen] columns: {cols}")
    ans_field = next((f for f in ANSWER_FIELD_CANDIDATES if f in cols), None)
    if ans_field is None:
        raise SystemExit(f"no answer field found in {cols}; set ANSWER_FIELD_CANDIDATES")
    print(f"[prep-gen] answer field: {ans_field}")

    rows, seen = [], set()
    for r in ds:
        if LANG_FIELD in cols and norm(r.get(LANG_FIELD)) not in ("", KEEP_LANG, "english"):
            continue
        ticket, answer = ticket_text(r), str(r.get(ans_field, "")).strip()
        if not ticket or len(answer) < MIN_ANSWER_CHARS:
            continue
        k = norm(ticket)
        if k in seen:
            continue
        seen.add(k)
        rows.append((ticket, answer))
    random.shuffle(rows)
    print(f"[prep-gen] usable (ticket, answer) pairs: {len(rows)}")
    if len(rows) < N_GOLD + 50:
        raise SystemExit(f"only {len(rows)} usable pairs; lower N_GOLD/N_TRAIN")

    gold = rows[:N_GOLD]
    train = rows[N_GOLD:N_GOLD + N_TRAIN]

    with open(os.path.join(DATA, "gold.jsonl"), "w") as f:
        for ticket, answer in gold:
            f.write(json.dumps({"messages": messages(ticket), "reference": answer}) + "\n")
    with open(os.path.join(DATA, "train_synth.jsonl"), "w") as f:
        for ticket, answer in train:
            f.write(json.dumps({"messages": messages(ticket, answer)}) + "\n")

    mix = [messages(t, a) for t, a in train]
    reh = []
    if os.path.exists(REHEARSAL_POOL):
        pool = [json.loads(x) for x in open(REHEARSAL_POOL) if x.strip()]
        random.shuffle(pool)
        reh = [p["messages"] for p in pool[:int(len(mix) * REPLAY_FRACTION)] if "messages" in p]
        print(f"[prep-gen] replay: added {len(reh)} rehearsal rows")
    else:
        print("[prep-gen] replay: no rehearsal pool; train_mix == train_synth")
    mixed = mix + reh
    random.shuffle(mixed)
    with open(os.path.join(DATA, "train_mix.jsonl"), "w") as f:
        for m in mixed:
            f.write(json.dumps({"messages": m}) + "\n")

    print(f"[prep-gen] wrote gold={len(gold)}, train_synth={len(train)}, train_mix={len(mixed)}")
    print("[prep-gen] review data/gold.jsonl, set confirmed: true in config.yaml, run pipeline_gen.py")


if __name__ == "__main__":
    main()
