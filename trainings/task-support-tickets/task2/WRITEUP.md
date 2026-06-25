# Task 2: third model family on a new corpus (Phi-4-mini + Jira)

Goal: show the pipeline is general by running it on a **third model family** and a **different corpus**
with no per-model code changes. It does, and the path there surfaced two model-specific wrinkles
(both fixed by config/model choice, not code) plus a memory limit worth recording.

- **Corpus:** the public Jira issues dataset (arXiv:2201.08368). One repository, IntelDAOS, 9,474
  issues, balanced into a 4-class issue-type task: Bug, Epic, Story, Sub-Task.
- **Model:** Phi-4-mini-instruct (3.8B, MIT). Different family/tokenizer/template from Qwen
  (pipeline2) and SmolLM3 (task1).

## Verdict: ACCEPT (2/2 seeds)

Attempt 1 (replay, lr 2e-4, 6 epochs), seeds [0, 1]: median task gain +0.206, both seeds passed.
Chosen `jira-issue-type-replay-s1`.

| axis | base | accepted (s1) | delta |
|------|------|---------------|-------|
| task macro-F1 | 0.376 | 0.658 | +0.282 |
| sentinel | 0.984 | 1.000 | +0.016 |
| reasoning | 1.000 | 1.000 | 0.000 |
| tools | 0.950 | 0.950 | 0.000 |

Per seed: s0 +0.130, s1 +0.282 (median +0.206), worst regression drop 0.000 on both. The base
model is already near-ceiling on the regression probes (a strong 3.8B), and fine-tuning left them
untouched while moving the task from 0.376 to 0.658. Clean pass, no adjusted re-run needed.

## What broke on the way, and why (the model-swap story)

### 1. Gemma 3 4B was the first pick, and it did not work out
Gemma 3 4B was chosen as the third family, but it hit two blockers:
- **Gated repo:** loading it raised `GatedRepoError / 401 Unauthorized` because Gemma is gated on
  HuggingFace (needs licence acceptance + an HF token).
- **Multimodal:** `gemma-3-4b-it` is `Gemma3ForConditionalGeneration` (vision + text), so it does
  not load cleanly through the pipeline's `AutoModelForCausalLM` text path.

Both are also off the "stick to permissively licensed models" guideline: Gemma is under the Gemma Terms
of Use, not an OSI-permissive licence.

### 2. Replaced with Phi-4-mini-instruct (MIT)
Phi-4-mini is **MIT-licensed, not gated, and text-only**, which removes all three issues at once
(no licence form, no token, no multimodal loading), and it is still a genuinely distinct third
family. One-line config change, no code edits.

### 3. Phi-4-mini's chat template is not training-compatible for assistant-only loss
Training then failed with:
`ValueError: The chat template is not training-compatible (missing prefix-preservation or
{% generation %} markers) ...`
TRL's `assistant_only_loss=True` masks the loss to just the assistant tokens, which needs the chat
template to mark the assistant span with `{% generation %}`. Qwen and SmolLM3 templates carry it
(or TRL auto-patches them); Phi-4-mini's does not. Fix: set `train.assistant_only_loss: false` (a
config flag, default true), so the run trains on the full sequence (prompt + answer). For
classification the label is a small fraction of each sequence, so this dilutes the task signal
slightly, but the gain here (+0.282 best seed) was strong regardless.

All three fixes are **config or model choice, never pipeline code**. See `../MODEL_NOTES.md` for the
running list of these per-model wrinkles.

## Honest caveat: early stopping did not run this time (OOM)

The log shows `[early-stop] epoch N: eval skipped (OutOfMemoryError ...)` for every epoch. The
clean-eval early stopping loads a **second bf16 copy** of the model each epoch to score the saved
adapter; a 3.8B second copy does not fit beside the 4-bit training model on a 14.6 GB T4. The
callback degraded gracefully (caught the OOM, logged "eval skipped", kept training) rather than
crashing, so the run completed as a **fixed 6-epoch** run (`best_epoch=None`). The final gate eval
runs after training with only one model resident, so the verdict is valid, but **adaptive length
did not engage** on this model.

To get adaptive length on a 3.8B/4B model you need either more VRAM than a free T4, or a lighter
in-loop signal (fewer probes, smaller gold subset, or evaluating the adapter without a separate
base copy). For this proof, fixed-epoch + the gate + the multi-seed median was enough; if you want
the early-stop trajectory too, run it where there is more GPU memory, or set
`early_stopping.enabled: false` to skip the failing per-epoch eval cleanly.

## Bottom line

Third model family (Phi-4-mini), new corpus (Jira issue types), gated and accepted on 2/2 seeds
(+0.282 best, no regression), with the pipeline code unchanged. The only adjustments were config:
the model id and one `assistant_only_loss` flag. That is the model-agnosticism claim holding across
three families (Qwen, SmolLM3, Phi-4-mini), with the per-model wrinkles documented rather than
hidden.
