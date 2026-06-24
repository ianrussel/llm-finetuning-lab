# Running `notebook-support-tickets.ipynb` on Kaggle

This notebook runs the Part 2 pipeline (config-driven training, two-axis gate, adaptive length,
multi-seed median) on a **new task and a new model**: queue routing on the Tobi-Bueck support
tickets, with **SmolLM3-3B**. It is the same pipeline as pipeline2 with only the config and the
data-prep changed. The local card is too small for a 3B model, so this runs on a free Kaggle GPU.

## Requirements

- A Kaggle account with a **verified phone number** (needed to turn Internet on).
- **GPU accelerator**: T4 x1 (16 GB) or P100. A 3B model will not fit on smaller.
- **Internet: ON.** The notebook downloads the model (SmolLM3-3B) from Hugging Face and the
  Tobi-Bueck dataset at runtime. Without internet both fail.

Set both under the right-hand panel: **Settings -> Accelerator -> GPU T4 x1**, and
**Settings -> Internet -> On**.

## What to attach (upload)

You do **not** need to upload the task data; the notebook builds it inline from Hugging Face.

Attach **one existing dataset** for the regression probes and the replay rehearsal pool:

- **`pipeline2-data`** (the same dataset your pipeline2 notebook uses). Add it via
  **Add Input -> Datasets -> your `pipeline2-data`**. The notebook expects it at
  `/kaggle/input/datasets/ianrusseladem/pipeline2-data` (the `PROBES` constant in cell 1). If your
  copy mounts at a different path (for example `/kaggle/input/pipeline2-data`), edit `PROBES` to
  match what shows under the Input panel.

That dataset must contain: `sentinel.jsonl`, `reasoning_probes.jsonl`, `tool_probes.jsonl` (the
60-probe regression sets) and `train_synth.jsonl` (used as the general rehearsal pool for replay).

## How to run

Run all cells top to bottom (**Run All**). The cells are:

1. **Deps** install (`transformers>=4.53` for SmolLM3, `trl`, `peft`, `bitsandbytes`, `datasets`).
2. **Config + helpers** (`CFG`, scoring functions).
3. **Build dataset** from Tobi-Bueck: filters to English, balances per queue, writes the gold and
   training files to `/kaggle/working/data`, and fills `LABELS`. It prints the dataset columns and
   the kept queue labels, check these look right before trusting the run.
4. **Training** (LoRA/QLoRA + clean-eval early stopping).
5. **Eval** (task macro-F1 on gold + the three regression probes).
6. **Gate + loop**: trains seeds `[0, 1]`, decides on the median, and does one adjusted re-run if
   the gate rejects. The final `ACCEPTED:` line names the chosen adapter.
7. **Download**: zips every trained adapter so you can pull them from the Output tab.

## What you get

- `result_*.json` for the base and each run (task + regression scores).
- `<run-name>.zip` per adapter under the Output tab.
- Console: per-epoch `[early-stop]` lines, per-seed `[gate]` verdicts, and the across-seeds median.

## Notes and gotchas

- **Memory.** SmolLM3-3B trains in 4-bit, but the clean-eval early stopping loads a **second bf16
  copy** of the 3B model each epoch to score the saved adapter. On a 16 GB T4 this fits but is the
  tightest moment. If you hit a CUDA OOM during an `[early-stop]` epoch: set
  `"early_stopping": {"enabled": False, ...}` in cell 1, or drop `"eval_batch"` to 4.
- **Swap the model with no code change.** Change one line in cell 1's `CFG`:
  `"base_model": "Qwen/Qwen2.5-0.5B-Instruct"`. Everything else is identical. Running both gives
  the same-task, two-model comparison.
- **Dataset schema.** Cell 3 picks the label field from `queue / type / priority / category` and
  prints the columns it sees. If the version you pull names things differently, adjust
  `LABEL_FIELD_CANDIDATES` / `TEXT_FIELDS` at the top of that cell.
- **Time.** Two seeds plus a possible adjusted re-run is up to ~4 training runs; budget roughly an
  hour of GPU. Reduce `"seeds"` to `[0]` for a quick single-run check.
