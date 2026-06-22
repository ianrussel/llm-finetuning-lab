# Second task, second model: proving the pipeline generalises

This folder answers the question "does the Part 2 pipeline only work for the one binary task and
the one model it was built on, or is it actually general?" It runs the **same** config-driven
pipeline (two-axis gate, adaptive training length, multi-seed median) on a **different corpus**
and a **different model family**, with no per-model code changes.

- **Different corpus:** [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
  real support tickets. The task is queue routing: given a ticket, predict its support queue (a
  closed multi-class label set), so the proven macro-F1 two-axis gate applies unchanged.
- **Different model:** [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (Apache 2.0),
  a different family with a different tokenizer and chat template from Qwen.

## The proof: no per-model code changes

The six pipeline files here are **byte-identical copies** of pipeline2's:

```
common_p2.py  gate.py  evaluate_from_config.py  train_from_config.py  early_stopping.py  pipeline.py
```

The only task-specific file is `prepare_data.py`, and the only model-specific change is one line
in `config.yaml` (`base_model`). Retargeting the model is a config edit, not a code edit, because
training and eval go through `tokenizer.apply_chat_template`, which each model supplies itself.

## Run order

```
# 0. deps (datasets, plus the training stack) in the project venv
pip install datasets

# 1. build the dataset from Tobi-Bueck (CPU, just downloads + processes)
../../.venv/bin/python prepare_data.py
#    review data/labels.txt and data/gold.jsonl

# 2. confirm the config (set confirmed: true in config.yaml), then run the gate loop
../../.venv/bin/python pipeline.py
```

`prepare_data.py` filters to English, balances per queue, holds out a sacred gold set, and
decontaminates gold vs train, writing the same `messages` shape the pipeline expects. The
regression probes (sentinel, reasoning, tools) are reused from Part 1 unchanged, since general
ability is task- and model-independent.

## Run on Kaggle

`kaggle_notebook/notebook-support-tickets.ipynb` runs the whole thing on a free GPU. It builds the
dataset inline from Tobi-Bueck (no upload needed) and reuses the existing `pipeline2-data` Kaggle
dataset for the regression probes and the rehearsal pool, so the only input you attach is that
dataset. Run top to bottom; the gate loop trains both seeds and decides on the median, same as
pipeline2.

## Swapping the model (the whole point)

To run the identical pipeline on Qwen instead of SmolLM3, change one line in `config.yaml`:

```yaml
base_model: Qwen/Qwen2.5-0.5B-Instruct   # was HuggingFaceTB/SmolLM3-3B
```

No code changes. Run `pipeline.py` again; it produces a separate set of `runs/result_*.json`.
That side-by-side (same task, same gate, two model families) is the generalisation evidence.

## Notes and caveats

- **Dataset schema is detected defensively.** `prepare_data.py` picks the label field from
  `queue / type / priority / category` and prints the columns it sees. If the version you pull
  names things differently, adjust `LABEL_FIELD_CANDIDATES` / `TEXT_FIELDS` at the top.
- **Memory.** SmolLM3-3B trains in 4-bit on a T4. The clean-eval early stopping loads a second
  bf16 copy of the 3B model each epoch to score the saved adapter, then frees it; on a 16 GB T4
  that fits but is tighter than the 0.5B Qwen run. If memory is short, lower `eval_batch` or set
  `early_stopping.enabled: false` to train the fixed `epochs` budget.
- **Replay.** `train_mix.jsonl` mixes the task with a general rehearsal sample drawn from Part 1's
  data if present; otherwise replay is off and `train_mix == train_synth` (the script says which).
- This reuses pipeline2's code by copying it. The honest tradeoff is duplication: a fix in
  pipeline2 does not propagate here automatically. For a learning repo that is acceptable and
  makes each task folder self-contained and runnable on its own.
