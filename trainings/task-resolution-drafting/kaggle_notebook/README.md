# Running `notebook-resolution-drafting.ipynb` on Kaggle

This runs the **open-ended** task: fine-tune a model to draft support replies, and score quality with
an **LLM judge** (a stronger model picks the better of the base vs fine-tuned answer per ticket).
Same config-driven pipeline, multi-seed median, and regression axis as the classification tasks; the
task axis is the judge **win-rate** instead of macro-F1.

## Requirements

- A Kaggle account with a **verified phone number** (for Internet).
- **GPU accelerator**: T4 x1 (16 GB) or P100.
- **Internet: ON.** The notebook downloads the trained model and the **judge (Qwen2.5-7B-Instruct,
  several GB)** from Hugging Face at runtime. Without internet both fail.

Set under the right panel: **Settings -> Accelerator -> GPU**, and **Settings -> Internet -> On**.

## What to attach (upload)

No task-data upload; the notebook builds it inline from Hugging Face. Attach **one dataset**:

- **`pipeline2-data`** (the same one your other notebooks use), for the regression probes
  (`sentinel.jsonl`, `reasoning_probes.jsonl`, `tool_probes.jsonl`) and the replay rehearsal pool
  (`train_synth.jsonl`). Add via **Add Input -> Datasets**. The notebook expects it at
  `/kaggle/input/datasets/ianrusseladem/pipeline2-data` (the `PROBES` constant in cell 1); if your
  copy mounts elsewhere, edit `PROBES`.

## How to run

Run all cells top to bottom:

1. **Deps**, **2. Config**, **3. Data** (Tobi-Bueck ticket -> reply).
2. **4. Eval / 5. Judge / 6. Gate loop**: trains seeds `[0, 1]`, generates each model's answers,
   loads the 7B judge once per stage to compare base vs candidate (both A/B orders, to cancel
   position bias), and accepts on the **median win-rate** across seeds. Final line is `ACCEPTED:`.
3. **7. Calibration (do this before trusting a verdict).** It writes `calibration.jsonl` pairing the
   base and chosen-candidate answers. Hand-label the `human` field (`a` / `b` / `tie`) on ~15 rows,
   then run `calibrate()`. Aim for about 0.8 agreement with your decisive judgments before believing
   the win-rate. An uncalibrated judge score is not evidence.
4. **8. Download** zips the adapters.

## What you get

- `result_*.json` for base and each run (answers + regression probe scores).
- `<run>.zip` adapters under the Output tab.
- Console: per-seed `[gate] win_rate=...`, the across-seeds median, and the `ACCEPTED:` adapter.

## Notes and gotchas

- **Calibrate first.** The win-rate is only trustworthy once the judge tracks your judgments
  (cell 7). This is Franz's rule, build the calibration habit before relying on the gate.
- **Memory.** The 0.5B base trains in 4-bit; the 7B judge loads in 4-bit only after the trained
  models are freed, and once per stage, so only one large model is resident at a time. Fits a T4.
- **Adaptive length is off for v1.** The clean-eval early stop scores macro-F1, which does not apply
  to generation; a ROUGE-vs-reference proxy is the planned addition. For now it trains fixed epochs
  and the judge gates the result.
- **Model-agnostic:** swap the trained model (`CFG["base_model"]`) or the judge
  (`CFG["judge"]["model"]`) with no code changes.
