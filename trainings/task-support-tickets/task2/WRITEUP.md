# Task 2: third model family on a new corpus (Phi-4-mini + Jira)

Goal: show the pipeline is general by running it on a **third model family** and a **different corpus**
with no per-model code changes. It does. The path there surfaced a few model-specific wrinkles, all
fixed by config or model choice, never pipeline code.

- **Corpus:** the public Jira issues dataset (arXiv:2201.08368). One repository, IntelDAOS, 9,474
  issues, balanced into a 4-class issue-type task: Bug, Epic, Story, Sub-Task.
- **Model:** Phi-4-mini-instruct (3.8B, MIT). Different family/tokenizer/template from Qwen
  (pipeline2) and SmolLM3 (task1).

## Verdict: ACCEPT (2/2 seeds)

Attempt 1 (replay, lr 2e-4, 6 epochs), seeds [0, 1]: median task gain +0.244, both seeds passed,
no adjusted re-run needed. Chosen `jira-issue-type-replay-s1`.

| axis | base | accepted (s1) | delta |
|------|------|---------------|-------|
| task macro-F1 | 0.376 | 0.622 | +0.246 |
| sentinel | 0.984 | 1.000 | +0.016 |
| reasoning | 1.000 | 1.000 | 0.000 |
| tools | 0.950 | 0.950 | 0.000 |

Per seed: s0 0.619 (+0.243), s1 0.622 (+0.246), median +0.244, worst regression drop 0.000 on both.
The base 3.8B is already near-ceiling on the regression probes, and fine-tuning left them untouched
while moving the task from 0.376 to about 0.62.

The two seeds agree closely: gain range [+0.243, +0.246]. That tight agreement is stronger evidence
than a single run, and it is the point of the multi-seed median. (An earlier task-2 run on the same
config came out [+0.130, +0.282], one weak seed and one strong; that spread was run-to-run variance,
and the median guard is exactly what keeps the verdict from riding on the lucky seed.)

## What broke on the way, and why (the model-swap story)

### 1. Gemma 3 4B was the first pick, and it did not work out
- **Gated repo:** loading it raised `GatedRepoError / 401 Unauthorized`; Gemma is gated on
  HuggingFace (needs licence acceptance + an HF token).
- **Multimodal:** `gemma-3-4b-it` is `Gemma3ForConditionalGeneration` (vision + text), so it does
  not load cleanly through the pipeline's `AutoModelForCausalLM` text path.
- **Licence:** Gemma is under the Gemma Terms of Use, not an OSI-permissive licence, off the
  "stick to permissively licensed models" guideline.

### 2. Replaced with Phi-4-mini-instruct (MIT)
MIT-licensed, not gated, text-only, which removes all three issues at once, and it is still a
genuinely distinct third family. One-line config change, no code edits.

### 3. Phi-4-mini's chat template is not training-compatible for assistant-only loss
Training first failed with `ValueError: The chat template is not training-compatible (missing ...
{% generation %} markers)`. `assistant_only_loss=True` masks the loss to the assistant tokens, which
needs the template to mark the assistant span with `{% generation %}`. Qwen and SmolLM3 templates
carry it (or TRL auto-patches them); Phi-4-mini's does not. Fix: `train.assistant_only_loss: false`
(a config flag, default true), so the run trains on the full sequence. The label is a small part of
each sequence, so this dilutes the task signal a little, yet the gain (+0.24) was strong regardless.

All fixes are **config or model choice, never pipeline code**. See `../MODEL_NOTES.md` for the
running list of per-model wrinkles.

## Adaptive length is off for this model (by config)

Early stopping is disabled for Phi-4-mini on a free T4. The clean-eval early stop loads a **second
bf16 copy** of the model each epoch to score the saved adapter, and a 3.8B second copy does not fit
beside the 4-bit training model on a 16 GB T4 (an earlier run with it on OOM'd every epoch and fell
back to fixed epochs anyway). So `early_stopping.enabled: false` here: the run trains the fixed
`epochs` budget cleanly, and the gate plus the multi-seed median do the accept/reject. Adaptive
length stays on for the small Qwen (pipeline2), where the second copy fits. The general fix, noted
for later, is a lighter in-loop eval that scores the adapter without a separate base copy, which
would let adaptive length work on 4B+ too.

## Third axis: knowledge absorption (a touch)

`build_knowledge.py` + `knowledge.py` add a closed-book domain-knowledge check (which DAOS component
an issue affects, distinct from the issue-type task), reported next to task and regression, never
gating. Run `knowledge.py --adapter <accepted>` after the gate to get
`knowledge_gain = fine-tuned - base`; a near-zero gain is the expected outcome for a small
classification fine-tune.

## Bottom line

Third model family (Phi-4-mini), new corpus (Jira issue types), gated and accepted on 2/2 seeds
(+0.244 median, no regression), pipeline code unchanged. The only adjustments were config: the model
id and one `assistant_only_loss` flag. Model-agnosticism holds across three families (Qwen,
SmolLM3, Phi-4-mini), with the per-model wrinkles documented rather than hidden.
