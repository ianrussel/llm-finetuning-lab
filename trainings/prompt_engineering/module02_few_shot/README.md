# Module 02: few-shot (in-context examples)

Sometimes describing the task isn't enough, you need to *show* it. Putting a few labelled
examples in the prompt (few-shot) teaches the model the exact output format and the decision
boundary, with no training involved. This is in-context learning.

## Zero-shot vs few-shot
- **Zero-shot:** just the instruction. Fast, but the model guesses the output shape and may
  add explanation, hedge, or phrase things inconsistently.
- **Few-shot:** the instruction plus 2 to 5 `input -> output` examples, then the real input.
  The examples pin the format and show how to handle edge cases.

## When to reach for few-shot
- you need a **consistent output shape** (a bare label, a fixed template),
- the task has a **specific style or convention** that's easier to show than describe,
- zero-shot is close but **inconsistent** run-to-run.

## Tips
- Keep examples **short, correct, and diverse** (cover the different cases/labels). A wrong
  example teaches the wrong thing.
- Make the example format **identical** to what you want back (here: `Label: <one word>`).
- 2 to 5 is usually enough; more examples cost context and have diminishing returns.

## Run it
```
python module02_few_shot/demo.py
```
It classifies a support message into {billing, technical, account} zero-shot vs few-shot.

## What to notice
The zero-shot reply often explains itself or phrases the label loosely; the few-shot reply is
a clean single label, because the examples demonstrated the exact form. Then try a genuinely
ambiguous message and watch how the examples shape the decision.

(This is also the cheap test from the playbook: if a few examples in the prompt make the
behavior reliable, you may not need to fine-tune at all.)
