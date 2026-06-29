# Per-model wrinkles (model-agnosticism notes)

The pipeline code is the same across models (the `*.py` modules are byte-identical between tasks).
What varies per model is NOT code, it is config, access, and chat-template properties. This doc
records the wrinkles hit so far so the next model swap is quick. Each fix is a config flag, an
access step, or a version bump, never a change to the pipeline logic.

## 1. Gated models (e.g. Gemma 3)

- **Symptom:** `GatedRepoError` / `401 Unauthorized` on `config.json`, "Access to model ... is restricted".
- **Cause:** the model is gated on HuggingFace.
- **Fix:** accept the licence on the model page, then authenticate. Local: `huggingface-cli login`
  or `HF_TOKEN` env. Kaggle: add `HF_TOKEN` in Add-ons -> Secrets. Colab: `login("hf_...")`.
- **Permissive alternative:** Phi-4-mini (MIT) and Granite (Apache) are NOT gated, so they download
  with no token. Prefer these to stay in the permissively-licensed set.

## 2. Multimodal models (e.g. gemma-3-4b-it)

- **Symptom:** model fails to load cleanly via `AutoModelForCausalLM` (it is
  `Gemma3ForConditionalGeneration`, a vision+text model).
- **Cause:** the 4B/12B/27B Gemma 3 models are multimodal; only the text-only path is wanted here.
- **Fix:** use a text-only model. Permissive options: Phi-4-mini-instruct (MIT), gemma-3-1b-it
  (text-only Gemma3), Granite (Apache), Qwen, SmolLM3.

## 3. Chat template without assistant-masking markers (e.g. Phi-4-mini)

- **Symptom:** `ValueError: The chat template is not training-compatible (missing prefix-preservation
  or {% generation %} markers) ...` when the trainer starts.
- **Cause:** `assistant_only_loss=True` (mask the loss to the assistant tokens) needs the chat
  template to mark the assistant span with `{% generation %}`. Qwen and SmolLM3 templates carry it
  (or TRL auto-patches them); Phi-4-mini's does not, and TRL will not guess.
- **Fix:** set `train.assistant_only_loss: false` (config flag, default true). The run then trains on
  the full sequence (prompt + answer) instead of answer-only.
- **Trade-off:** for classification the label is a small fraction of each sequence, so full-sequence
  loss dilutes the task signal a little. If task gain comes out weak, the adjusted re-run (lower lr)
  or a couple more epochs usually recovers it. For long open-ended answers the dilution is small.

## 4. Reasoning models with a default think block (e.g. SmolLM3)

- **Symptom:** the base model scores artificially low on the sentinel probe (e.g. 0.23 instead of
  ~0.9), and the in-loop sentinel is noisy enough to trip the early-stop regression guard.
- **Cause:** the model emits a `<think> ... </think>` block by default; a short probe budget
  (`max_new=32`) is consumed by thinking before the answer appears, so it scores as a miss.
- **Fix:** give the sentinel probe enough tokens. It is set to `max_new=128` in the eval and
  early-stop probe lists, so the answer survives the think block. Reasoning (256) and tools (128)
  already had room.

## 5. transformers version

- Newer model families need a recent `transformers`: Gemma 3 needs `>= 4.50`, Phi-4 `>= 4.49`.
  The notebooks pin `transformers>=4.50` to cover both. A too-old version shows up as an unknown
  architecture or a missing config class.

## 6. Memory on a free T4 (3B and 4B models)

- **Symptom:** per-epoch `[early-stop]` either CUDA-OOMs or throws a tensor-size mismatch.
- **Cause:** the in-loop eval needs to run the model each epoch, and neither mode fits a 3.8B/4B on a
  free 16 GB T4: `eval_mode: clean` loads a second bf16 copy (3B ~6 GB, 4B ~8 GB) beside the 4-bit
  trainer and OOMs; `eval_mode: resident` reuses the in-training model (no second copy) but batched
  `generate()` on the live QLoRA model throws a size mismatch (observed on Phi-4: the in-training
  model is not a clean inference model). Both were tried on Phi-4-3.8B; both failed, and the callback
  degraded gracefully to fixed epochs each time.
- **Practical fix:** for 3.8B+ on a free T4, set `early_stopping.enabled: false` and rely on the gate
  + multi-seed median (reliable; it accepted 2/2). Adaptive length is worth it on small models (the
  0.5B Qwen, `eval_mode: clean`) where the second copy fits and matches the gate exactly.
- **To actually get adaptive length on big models** you need more VRAM, or an in-loop signal that
  needs neither a second copy nor `generate()` on the live model, e.g. held-out eval **loss** (a
  forward pass) for the task-plateau signal. Not built yet; the eval modes (`clean`/`resident`) are
  the two `early_stopping.eval_mode` options today.

## Summary

Model-agnosticism holds at the code level: the same pipeline trained Qwen 0.5B, SmolLM3-3B, and
Phi-4-mini-3.8B with no logic changes. The per-model differences above are all config, access, or
version, which is exactly what "config is the single source of truth" is supposed to absorb.
