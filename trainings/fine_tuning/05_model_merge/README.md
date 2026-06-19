# Module 5: Model merge

Goal: merge a fine-tuned model with its base using MergeKit and observe the effect.

This reuses the module 2/3 support-ticket classifier as the fine-tune, and the same 6
held-out eval cases for scoring, so the effect of merging is measured on a task I already
understand.

## The idea

Merging combines the weights of two models into one, no extra training. The simplest
form (and the one here) is a weighted average of the base and a fine-tune:

    merged = alpha * fine_tuned + (1 - alpha) * base

- alpha = 0 is the pure base (general, no task skill).
- alpha = 1 is the pure fine-tune (task skill, but may have drifted from general use).
- in between is a blend. Merging back toward the base is a way to keep most of the
  task ability while pulling some general behavior back, a dial, not an on/off switch.

## Why there is an extra first step

MergeKit merges full model weights, not LoRA adapters. My fine-tunes are adapters, so I
first bake an adapter into the base (peft merge_and_unload) to get a standalone full
model, then merge that with the base.

## Setup (once)

```bash
.venv/bin/python -m pip install mergekit
```

(Heads up: installing mergekit downgrades a few shared packages like accelerate and
huggingface_hub. The rest of the stack still works, but if something acts up, that is
where to look.)

## Pipeline

```bash
aipy prepare_finetuned.py   # adapter -> ./finetuned-full (a standalone fine-tuned model)
aipy merge.py               # linear-merge base + fine-tune at alpha 0.25 / 0.50 / 0.75
aipy evaluate.py            # score base, blends, and fine-tune; show a general prompt too
```

merge.py writes a MergeKit config per alpha (merge_25.yml etc.) and runs mergekit-yaml,
producing ./merged-25, ./merged-50, ./merged-75. Everything runs on CPU; the 0.5B merge
is fast and does not need the GPU.

## Files

- prepare_finetuned.py: folds module 3's LoRA adapter into the base, saves ./finetuned-full.
- merge.py: builds the MergeKit configs and runs the linear merges.
- evaluate.py: scores every model (valid JSON / category / exact) and prints each one's
  answer to an off-task general prompt.
- common.py: shared SYSTEM prompt, eval helpers, and the general prompts.
- data/eval.jsonl: the 6 held-out cases (same as module 2/3).

## What to look for

- A trade-off curve. As alpha rises from 0 to 1, the task scores should climb. The
  general-prompt answers may stay sensible at low alpha and drift as the fine-tune
  dominates (or the reverse, depending on how much the fine-tune narrowed the model).
- The point of the milestone is to see that merging is a cheap, training-free way to
  blend capabilities, and to feel the tension between task skill and general ability.
- Our fine-tune is a tiny classifier that already kept JSON format, so the effect may be
  subtle. If you want a louder effect, merge an adapter that changed the model more, or
  try a different merge_method (task_arithmetic, ties, dare_ties).

## Make it my own

- Change ALPHAS in merge.py to scan more points (e.g. 0.1 ... 0.9).
- Swap merge_method in the generated config to task_arithmetic or ties (those need a
  base_model: line in the config).
- Point prepare_finetuned.py at a different adapter (module 1 or 2) and compare.
