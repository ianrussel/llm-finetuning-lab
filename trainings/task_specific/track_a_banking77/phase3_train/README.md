# Phase 3: train the LoRA specialist

Trains a LoRA/QLoRA adapter on the set built in phases 1-2, on the same
Qwen2.5-0.5B-Instruct base the evaluator scores. One script, `train.py`, run from
the `track_a_banking77` folder.

## Two things this phase gets right

- **Assistant-only loss.** Training happens on the label tokens only, not on the
  ~420-token system prompt and the user query. TRL masks everything except the
  assistant turn (the Qwen2.5 chat template marks it), so the short label is the
  whole learning signal. This is the fix for module 1's free-completion bug, where
  training on the full text taught the model to echo the prompt.
- **No truncated labels.** The label is the last token in every row, so a too-low
  length cap would cut exactly what we want to learn. `--max-len` defaults to 768,
  above the longest row (~637 tokens), and the script prints any row that would
  still overflow so it never happens silently.

## The three-condition comparison

The point of the track is to isolate what the synthetic data adds, so train two
adapters from the same base and compare both against the recorded base baseline:

```
# control: seed only (385 real rows)
../../../.venv/bin/python phase3_train/train.py --data data/seed.jsonl --name seed

# the real run: seed + kept synthetic
../../../.venv/bin/python phase3_train/train.py --data data/train_synth.jsonl --name seed-synth
```

Smoke-test the wiring first on a tiny slice (one quick epoch, a few rows):

```
../../../.venv/bin/python phase3_train/train.py --data data/train_synth.jsonl \
    --name smoke --limit 16 --epochs 1
```

## What it writes

`phase3_train/lora-<name>/` holds the adapter (small, not the full model), the
tokenizer, and `train_config.json` (base model, data, epochs, LR, LoRA rank/alpha,
final train loss) so the run is reproducible out of the session.

## Knobs worth touching

- `--epochs` (default 3). The label is tiny, so loss drops fast; if the gold
  number looks undertrained, add epochs before anything else.
- `--lr` (default 2e-4), `--rank` / `--alpha` (default 16 / 32), `--batch` /
  `--grad-accum` (default 2 / 8, effective batch 16). Lower the batch first if you
  hit an out-of-memory error on a small GPU.

## Then evaluate (Phase 4)

Re-run the identical gold + sentinel sets with each adapter, then line all three
conditions up against the base baseline with `eval/compare.py`:

```
../../../.venv/bin/python eval/evaluate.py --name seed       --adapter phase3_train/lora-seed
../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
../../../.venv/bin/python eval/compare.py
```

`compare.py` writes `data/comparison.json` and prints the before/after on both
axes plus the per-intent F1 movement the synthetic data is responsible for. The
recorded three-condition result lives in the track README.
