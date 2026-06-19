# Track B: corpus-grounded QA specialist

The track that mirrors the real scenario: take a document corpus, turn it into a
grounded task, and train a small specialist that answers ONLY from the supplied
context and abstains when the answer is not there. Same end-to-end discipline as
Track A (sacred gold, no fabrication, decontaminate, two-axis eval, iterate), on a
harder task where the failure mode that matters is hallucination.

## Layout

Organised by phase, with shared pieces at the track root.

```
track_b_squad_grounded/
  common.py            task contract + helpers (RAFT assembly, abstention, EM/F1)
  data/                shared artifacts: seed, gold, passages, generated, results
  phase1_seed/         build_seed.py        Phase 1: define task + build seed/gold
  phase2_synthetic/    sdg.py qgen.py judge.py filter.py
  phase3_train/        train.py             Phase 3: train the grounded LoRA
  phase5_iterate/      error_analysis.py targeted.py build_v2.py   Phase 5: iterate
  eval/                evaluate.py compare.py sentinel.jsonl   two-axis scorer (Phase 1 + 4)
```

Run everything from the `track_b_squad_grounded` folder so the `../../../.venv/bin/python`
prefix and the script paths line up.

## The corpus and the task

Corpus is `rajpurkar/squad_v2`: Wikipedia passages with crowd-written questions,
including a large block of deliberately **unanswerable** questions (on-topic, but
the passage does not contain the answer). That unanswerable block is what makes
this a grounding task and not just QA: the specialist has to know when to say "I
do not know" rather than invent something.

- **input**: a context block of one or more passages, and a question.
- **output**: the answer as a short exact span copied from the context, OR the
  fixed abstention string `not in the context` when the answer is not present.
- **grounding rule**: answer only from the provided passages, never from outside
  knowledge.

### RAFT

Every training and eval row is assembled RAFT-style: the oracle passage (the one
holding the answer) is mixed with distractor passages from other articles and
shuffled, so the model learns to find the answer among irrelevant context and to
abstain when none of the passages contain it. Unanswerable rows are all-distractor
by construction. Default is 2 distractors per row (`N_DISTRACTORS`).

## Success metric

Scored on `data/gold.jsonl` only, split by question type:

- **answerable**: SQuAD exact-match and token-F1 against the gold answers.
- **unanswerable**: abstention accuracy (did it correctly say `not in the context`)
  and the hallucination rate (the share it answered anyway).
- **grounded_score**: one combined number, the fraction of all gold rows handled
  correctly (answerable above an F1 bar, or unanswerable correctly abstained).
- **faithfulness (optional)**: `evaluate.py --judge` adds an LLM-as-judge pass that
  checks each answered row against its oracle passage, the doc's faithfulness axis,
  catching plausible-but-unsupported answers that string overlap would miss.

The EM/F1/abstention metrics are deterministic and need no model, so eval is
reproducible; the judge pass is the extra faithfulness layer when Ollama is up.

## The pipeline (run order)

```
# Phase 1: build the seed + sacred gold from SQuAD v2, and the passage pool
../../../.venv/bin/python phase1_seed/build_seed.py

# Baseline: the untuned base model on both axes (the "before")
../../../.venv/bin/python eval/evaluate.py --name base

# Phase 2: document-grounded synthetic data (Ollama up; in order)
../../../.venv/bin/python phase2_synthetic/qgen.py      # passages -> QA candidates
../../../.venv/bin/python phase2_synthetic/judge.py     # faithfulness gate
../../../.venv/bin/python phase2_synthetic/filter.py    # dedup + decontaminate + RAFT assemble

# Phase 3: train the two adapters (GPU)
../../../.venv/bin/python phase3_train/train.py --data data/seed.jsonl --name seed
../../../.venv/bin/python phase3_train/train.py --data data/train_synth.jsonl --name seed-synth

# Phase 4: re-measure each and line them up
../../../.venv/bin/python eval/evaluate.py --name seed       --adapter phase3_train/lora-seed
../../../.venv/bin/python eval/evaluate.py --name seed-synth --adapter phase3_train/lora-seed-synth
../../../.venv/bin/python eval/compare.py

# Phase 5: iterate on the dominant failure mode
../../../.venv/bin/python phase5_iterate/error_analysis.py
../../../.venv/bin/python phase5_iterate/targeted.py
../../../.venv/bin/python phase5_iterate/build_v2.py
../../../.venv/bin/python phase3_train/train.py --data data/train_synth_v2.jsonl --name seed-synth-v2
../../../.venv/bin/python eval/evaluate.py --name seed-synth-v2 --adapter phase3_train/lora-seed-synth-v2
../../../.venv/bin/python eval/compare.py --names base seed seed-synth seed-synth-v2 \
    --effect-pair seed-synth seed-synth-v2
```

## Using your own corpus instead of SQuAD

`build_seed.py` pulls SQuAD v2, which is reproducible and ships verified gold QA.
To run the track on a real documentation / FAQ corpus instead, use the drop-in
`phase1_seed/build_seed_from_docs.py`. It ingests a folder of `.md` / `.txt` /
`.html` / `.rst` files, chunks them into passages, and reserves a held-out set of
documents for gold (split by document, so generation cannot leak into eval):

```
../../../.venv/bin/python phase1_seed/build_seed_from_docs.py --docs-dir /path/to/docs
```

It writes the same `data/passages.jsonl`, `data/seed.jsonl` and `data/gold.jsonl`
the rest of the pipeline expects, so Phases 2 to 5 run unchanged. The one extra
discipline: your docs have no verified answers, so the script DRAFTS gold/seed with
the local model and marks every row `"verified": false`. You must open
`data/gold.jsonl`, check each answer against its oracle passage, fix or drop the
bad rows, and set `verified: true` before trusting any eval number, a model graded
on machine-drafted gold measures nothing. Pass `--no-generate` to get just the
passage pool and write the gold QA entirely by hand.

Where to find a corpus: an open-source project's `docs/` folder (Markdown, the
easiest), your own product docs or a help-center export, a public FAQ page set, or
a docs/QA dataset on Hugging Face. Pick a domain you can judge, mind the licence,
and keep it small, even 20 to 100 documents chunk into hundreds of passages.

## Disciplines enforced (same spine as Track A)

- gold comes from the validation split, seed and synthetic from train, so they are
  disjoint by split; a normalized-question decontamination check drops any seed row
  that collides with gold and asserts zero overlap.
- Phase 2 passages exclude every gold oracle passage, so generated QA cannot be
  grounded in a gold passage; `filter.py` also drops any candidate whose question
  near-matches a gold question.
- the synthetic unanswerable questions are the abstention training signal, and the
  judge confirms each is genuinely unanswerable before it is kept, so the model is
  not taught to abstain on questions that were actually answerable.
- training takes loss on the answer tokens only (`assistant_only_loss`), and
  `--max-len` is set high so the RAFT context is never silently truncated.

## What is expected to move

The base Qwen2.5-0.5B-Instruct tends to ignore the grounding instruction: it
answers from memory and rarely abstains, so the unanswerable block is where it
bleeds (high hallucination). The seed fine-tune should teach the format and some
abstention; the document-grounded synthetic data, especially the generated
unanswerable questions, is the lever expected to cut hallucination and lift
grounded_score, with the regression sentinel held flat. Phase 5 then targets
whichever mode is still weak (hallucination or answerable accuracy).

## Test the trained model

Ask the grounded model questions about a passage (it answers only from the context, or abstains):

```
../../../.venv/bin/python run_trained.py
# question> how long does the reef stretch?      (type 'ctx: <passage>' to change the context)
```

Flags: `--adapter <path>`, or `--context "..." --question "..."` for a one-shot run.
