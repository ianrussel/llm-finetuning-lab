# Module 2: Dataset from scratch

Goal: hand-build a tiny task-specific dataset in instruction/JSONL format and train
on it, so the data shaping step is something I actually do by hand.

## The task

Classify a one-line support message into strict JSON and nothing else:

```json
{"category": "billing|technical|account|general", "priority": "low|medium|high"}
```

The base model does not do this on its own (it explains and chats instead of emitting
clean JSON), so the before/after is easy to see. That is the point: pick a narrow task
the base model is bad at.

## What "hand-build a dataset" means here

A dataset for this is just a list of input -> desired-output pairs, one per line, in the
chat message format the trainer understands. Two decisions do all the work:

1. Fix the output format exactly. Always the same JSON shape, same key order, same
   allowed values, no prose.
2. Apply one consistent labeling rule so examples do not contradict each other:
   - high: user is blocked, locked out, or wrongly charged
   - medium: something broken with a workaround, or a money question to resolve
   - low: general question, how-to, or praise

If the examples are inconsistent, the model learns inconsistency. That is the whole
lesson of this module.

## Files

- data/train.jsonl: 24 hand-written examples in the conversational "messages" format
  (system + user + assistant). The trainer applies the chat template automatically.
- data/eval.jsonl: 6 held-out messages the model never trains on, used to measure
  before vs after. These have no system line; the test scripts add it.
- train.py: LoRA fine-tune that reads data/train.jsonl and writes the adapter to
  ./lora-out. 8 epochs because the dataset is tiny.
- test_base.py: the "before". Runs the untuned base model over data/eval.jsonl and
  prints a score (valid JSON count, exact match count).
- test_adapter.py: the "after". Same eval and scoring, but with the trained adapter.

## One line of the dataset, explained

```json
{"messages": [
  {"role": "system",    "content": "You are a support ticket classifier. Reply with ONLY a JSON object ..."},
  {"role": "user",      "content": "I was charged twice for my subscription this month."},
  {"role": "assistant", "content": "{\"category\": \"billing\", \"priority\": \"high\"}"}
]}
```

The system line states the task, the user line is the input, the assistant line is the
exact output I want the model to produce. The model trains to reproduce that assistant
line given the system + user.

## How to run

GPU (Kaggle T4 etc.) is the intended place to train. From this folder:

```bash
# before: untuned base model on the held-out set
python test_base.py

# train on the hand-built dataset
python train.py

# after: same held-out set, now with the adapter
python test_adapter.py
```

Compare the two summary lines. I expect the base model to score low on valid JSON and
exact match, and the tuned adapter to score much higher. That gap is the before/after I
did not get in module 1.

## Make it my own

- Swap the task entirely (different categories, a different output format, a made-up
  domain). Keep the format fixed and the labels consistent.
- Add more examples by hand. Just append lines to data/train.jsonl in the same shape.
  More coverage usually means cleaner outputs.
- If outputs are messy, the fix is almost always the data: a contradictory label, a
  format that drifts between examples, or too few examples for some category.
