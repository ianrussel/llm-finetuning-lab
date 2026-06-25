# Task 2: third model family, different corpus, same pipeline

Task 1 proved the pipeline on SmolLM3-3B + Tobi-Bueck support tickets. Task 2 pushes
model-agnosticism to a **third family** on a **different corpus**, with the same code:

- **Corpus:** the public Jira issues dataset (arXiv:2201.08368, Montgomery et al., "An Alternative
  Issue Tracking Dataset of Public Jira Repositories"). Task: predict an issue's **type** (Bug /
  Improvement / New Feature / Task / ...) from its summary + description. Closed label set, so the
  proven macro-F1 two-axis gate applies unchanged.
- **Model:** [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it), a different tokenizer and
  chat template again.

The six `*.py` files here are **byte-identical** to task1's; only `config.yaml` and
`prepare_data.py` differ. Retargeting the model is a one-line config change.

## Before you run: two setup steps this task needs

1. **Get the Jira dataset.** Unlike Tobi-Bueck, it is not a one-line HuggingFace load; it ships as
   large per-repository dumps (Zenodo, MongoDB/JSON). Convert one or more repositories' issues into
   a JSONL at `data/jira_issues.jsonl`, one issue per line, keeping at least the type, summary, and
   description fields (flat or under a nested `fields` object, both are handled). Then point `SOURCE`
   in `prepare_data.py` at it (or set `SOURCE` to a HF mirror id if you use one). The script prints
   the detected columns and a sample on first run so you can confirm the schema and adjust the field
   paths if needed.
2. **Gemma access.** Gemma 3 is gated on HuggingFace: accept the licence on the model page and set an
   `HF_TOKEN`. It also needs a recent `transformers` (>= 4.50).

## Run order

```
pip install -U "transformers>=4.50" datasets
../../../.venv/bin/python prepare_data.py        # build data/ from your Jira JSONL
#   review data/labels.txt and data/gold.jsonl, set confirmed: true in config.yaml
../../../.venv/bin/python pipeline.py            # same gate loop, now on Gemma 3 4B + Jira
```

Per-model wrinkles hit while swapping families (gating, multimodal, chat-template assistant masking,
reasoning-model probe budget, memory) are documented once in [../MODEL_NOTES.md](../MODEL_NOTES.md).
This task uses Phi-4-mini (MIT, ungated) with `assistant_only_loss: false` because its chat template
lacks the `{% generation %}` markers TRL needs; see that doc for the why.

## Third axis: knowledge absorption (a touch, not a gate)

Next to the task and regression axes, a light check of whether the fine-tune absorbed DAOS *domain*
knowledge. The Jira `components` field (Control Plane, Erasure Code, Rebuild, ...) is a domain fact
distinct from the task label (issue type), so it does not just re-measure the task.

```
../../../.venv/bin/python build_knowledge.py                 # held-out, decontaminated Q&A set (CPU)
../../../.venv/bin/python knowledge.py --adapter runs/jira-issue-type-replay-s1   # closed-book, GPU
```

- `build_knowledge.py` turns issues that are NOT in train/gold into ~40 closed-book questions
  ("which component does this issue affect?"), decontaminated against the exact training text, scored
  with `any`-match (naming any correct component counts).
- `knowledge.py` scores base vs fine-tuned closed-book (no retrieval) and reports
  `knowledge_gain = fine-tuned - base` into `runs/knowledge.json`. Positive means some domain
  knowledge stuck; near-zero is the expected, acceptable outcome for a small classification
  fine-tune. It is **reported alongside the gate, never used to accept or reject** (it is a touch,
  not the main goal). The same pair of scripts works for any task by pointing at its corpus.

## Notes and caveats

- **Licence.** Gemma is under the Gemma Terms of Use, not an OSI-permissive licence like the
  Apache/MIT models preferred for this work. Chosen here per request. To stay fully permissive, swap
  `base_model` to `microsoft/Phi-4-mini-instruct` (MIT) or an `ibm-granite` model (Apache), no code
  change.
- **Memory.** Gemma 3 4B trains in 4-bit, but the clean-eval early stopping loads a second bf16 copy
  (~8 GB) each epoch next to the 4-bit training model; on a 16 GB T4 this is tight. If it OOMs, set
  `early_stopping.enabled: false` or lower `eval_batch`.
- **Probe path.** task2 sits one level deeper than the original folder, so `probes_dir` is
  `../../task-dataset-generation-pipeline1/eval` (note the `../../`). The same Part 1 regression
  probes guard this run.
- **What this proves:** a third model family on a new corpus, gated and accepted with no per-model
  code changes, alongside Qwen (pipeline2) and SmolLM3 (task1) — model-agnosticism across 3+.
