# Online notebooks (Kaggle / Colab)

Self-contained notebooks for running the **training and evaluation** of each track
on a free cloud GPU (Kaggle T4, 16 GB), so you are not limited by the 4 GB local
laptop GPU. Each notebook embeds all the helper code and the regression sentinel,
so the only thing you upload is the data.

```
kaggle_notebooks/
  track_a_banking77_kaggle.ipynb      Track A: train + eval banking77 intents
  track_b_squad_grounded_kaggle.ipynb Track B: train + eval grounded QA
  reasoning_sft_kaggle.ipynb          Part B: reasoning-chain distillation SFT (GSM8K)
```

## The hybrid workflow

Generation stays local (Phase 2 needs Ollama); only the GPU-heavy training moves
to the cloud. Two small files cross each way:

```
local: generate data  ->  upload data/*.jsonl as a Kaggle Dataset
cloud: train + eval    ->  download lora-*.zip + result_*.json
local: (optional) re-run eval / iterate
```

## How to run on Kaggle

1. **Make the data dataset.** Each notebook expects one Kaggle Dataset:
   - Track A -> name it `track-a-data`, containing `labels.txt`, `gold.jsonl`,
     `seed.jsonl`, `train_synth.jsonl` (from `track_a_banking77/data/`).
   - Track B -> name it `track-b-data`, containing `seed.jsonl`, `gold.jsonl`,
     `train_synth.jsonl` (from `track_b_squad_grounded/data/`).

   Quick way to gather the files to upload (run locally):
   ```bash
   # Track A
   mkdir -p /tmp/track-a-data && cp track_a_banking77/data/{labels.txt,gold.jsonl,seed.jsonl,train_synth.jsonl} /tmp/track-a-data/
   # Track B (after Phase 2 has produced train_synth.jsonl)
   mkdir -p /tmp/track-b-data && cp track_b_squad_grounded/data/{seed.jsonl,gold.jsonl,train_synth.jsonl} /tmp/track-b-data/
   ```
   Then on kaggle.com: Datasets -> New Dataset -> upload that folder.

2. **Upload the notebook.** Kaggle: Create -> Notebook -> File -> Upload Notebook,
   pick the `.ipynb`. Or Colab: File -> Upload notebook.

3. **Turn on the GPU.** Kaggle: Settings -> Accelerator -> **GPU T4**. Then add your
   dataset via the right panel (Add Input) so it mounts at `/kaggle/input/<slug>/`.

4. **Run all.** The notebook trains `seed` and `seed-synth`, evaluates base vs seed
   vs seed-synth on both axes, prints the comparison table, and zips the adapters.

5. **Bring the results home.** Download `lora-seed.zip`, `lora-seed-synth.zip` and
   `result_*.json` from the Output tab. Unzip each adapter into
   `track_<x>/phase3_train/lora-<name>/` if you want to re-run the local evaluator.

## Notes

- **Same recipe as local.** The notebooks mirror `phase3_train/train.py` and
  `eval/evaluate.py` (same base model, LoRA config, assistant-only loss, metrics),
  so the cloud numbers are directly comparable to the local ones.
- **Batch sizes are tuned for a 16 GB T4.** Track A uses batch 4 (`max_len 768`);
  Track B uses batch 1 with `max_len 1536` for the long RAFT contexts, with
  `grad_accum` set so the effective batch stays 16. The binding constraint is the
  loss-step logits tensor (sequence x Qwen's ~152k vocab, in fp32), which is large
  for long contexts regardless of GPU, so batch is the main OOM lever. Raise it on
  a bigger GPU; if you still OOM, lower `max_len` or batch further.
- **Bigger base model.** With 16 GB you can swap `BASE_MODEL` to
  `Qwen/Qwen2.5-1.5B-Instruct` or `-3B-Instruct` in the generation-helpers cell for
  a likely stronger result. If you do, also re-measure the base baseline so the
  before/after stays honest.
- **Colab** works the same way; upload the data to Google Drive (or use
  `files.upload()`) and point `DATA` at it instead of `/kaggle/input/...`.
- These notebooks cover Phases 3 and 4 (train + eval). Phase 2 generation and
  Phase 5 iteration stay local with Ollama, per each track's README.
