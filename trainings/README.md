# Track A: banking77 intent-classification specialist

The warm-up track from the corpus-to-specialist deep dive: take banking77, build
a small verified seed and a sacred gold set, expand the seed with synthetic data,
train a LoRA, and measure it honestly on two axes.

## Layout

The code is organised by phase. Shared pieces stay at the track root.

```
track_a_banking77/
  common.py            shared task contract + helpers (every phase imports this)
  data/                shared artifacts: labels, seed, gold, results, predictions
  phase1_seed/         build_seed.py        Phase 1: define task + build seed/gold
  phase2_synthetic/    sdg.py paraphrase.py evolve.py judge.py filter.py
  eval/                evaluate.py sentinel.jsonl   two-axis scorer (Phase 1 + 4)
```

Run everything from the `track_a_banking77` folder so the relative venv path
(`../../../.venv/bin/python`) and the script paths below line up. `common.py` and
the `data/` files resolve by absolute path, so the working directory only matters
for the venv prefix.

## The task

banking77 intent classification. The corpus is `PolyAI/banking77`: real online
banking customer messages, each labelled with one of 77 fine-grained intents.
The intents are deliberately confusable (for example `card_about_to_expire` vs
`card_arrival`, or `top_up_failed` vs `top_up_reverted`), so a small base model
has real headroom to improve. That is what makes the before/after measurable.

## Input/output contract

- input: one customer message, a short free-text string.
- output: exactly one intent label from the 77-label closed set, copied
  verbatim, with nothing else (no quotes, no punctuation, no explanation).
- format on disk: the repo's conversational `messages` shape, one JSON object
  per line: a system message holding the instruction and the full label list, a
  user message with the customer query, an assistant message with the bare
  label. This matches modules 2, 3 and 5 so the existing trainer and scoring
  habits carry over.

The 77 labels are written to `data/labels.txt` in a fixed order. Two of them are
quirks of the source dataset and are kept verbatim because the model has to
reproduce them exactly: `Refund_not_showing_up` is capitalised, and
`reverted_card_payment?` ends in a question mark.

## Success metric

- accuracy on the gold set (fraction of queries given the exact right label).
- macro-F1 across the 77 intents (averaged per class, so a few well-covered
  intents cannot hide weakness on the rare ones). Macro is the headline number
  because the classes are many and easy to confuse.

Both are scored as an exact string match between the assistant output and the
gold label, measured on `data/gold.jsonl` only.

## Phase 1: what it produced

`phase1_seed/build_seed.py` loads banking77 and writes three files into `data/`:

- `labels.txt`: the 77-intent output vocabulary.
- `gold.jsonl`: the sacred held-out evaluation set. 1540 rows, 20 per intent,
  sampled from the dataset's test split. Nothing derived from this set may ever
  enter training.
- `seed.jsonl`: a small, balanced, verified training seed. 385 rows, 5 per
  intent, sampled from the train split.

The seed is intentionally tiny (5 examples per intent). The LIMA idea is that a
specialist needs surprisingly few real examples once the task is well defined;
keeping the seed small also leaves clear headroom for Phase 2 synthetic data to
show a measurable effect when we compare base vs seed-only vs seed-plus-synthetic.

## Disciplines already enforced here

- The gold set is reserved first and comes from the test split, while the seed
  comes from the train split, so the two are disjoint by construction.
- Decontamination runs before sampling the seed: any train query that also
  appears in gold is dropped (banking77 really does share a handful of identical
  queries across splits). This run dropped 5 such leaks plus 4 in-pool
  duplicates.
- A final assertion confirms zero normalised-query overlap between seed and
  gold. The independent check also confirms all 77 intents are covered and the
  per-intent counts are exactly balanced.

## Reproduce

Run with the project venv (plain `python` is the wrong interpreter on this
machine):

```
../../../.venv/bin/python phase1_seed/build_seed.py
```

The run is deterministic (fixed seed), so it reproduces the same split. Knobs at
the top of the script: `SEEDS_PER_INTENT`, `GOLD_PER_INTENT`, `SEED`.

## Baseline (the "before" for both axes)

`eval/evaluate.py` scores a model on the same two sets every time and writes
`data/result_<name>.json`. Run the untuned base model to capture the
pre-training baseline:

```
../../../.venv/bin/python eval/evaluate.py --name base
```

Base `Qwen/Qwen2.5-0.5B-Instruct` on the full 1540-row gold set:

- accuracy: 0.218
- macro-F1: 0.203
- valid-label rate: 0.731 (only 73% of outputs are even a real intent)
- regression sentinel: 11/12

That low base score is the point: it leaves clear headroom for the seed-only and
seed-plus-synthetic fine-tunes to show a real delta, measured against this exact
baseline. Saved to `data/result_base.json`.

Scoring notes: the output is matched to a label leniently (exact first-line match,
then a verbatim substring, longest wins), which cannot inflate the score but does
give the base model a fair read. `valid-label rate` reports how often it produced
a real intent at all. The sentinel is a small fixed offline stand-in for an MMLU
slice; swap in lm-evaluation-harness later if a heavier regression check is wanted.

## Phase 2: synthetic data generation

Phase 2 turns the 385-row seed into a larger, more varied training set without
fabricating facts and without contaminating the gold set. It runs as a four-step
pipeline on a LOCAL model (Ollama, same privacy-preserving choice as module 3),
so nothing leaves the machine. `phase2_synthetic/sdg.py` holds the shared Ollama
connection.

Setup once:

```
ollama pull qwen2.5:3b-instruct
ollama serve     # if it is not already running
```

Then, in order:

```
../../../.venv/bin/python phase2_synthetic/paraphrase.py   # pass 1: reword each seed, same intent
../../../.venv/bin/python phase2_synthetic/evolve.py       # pass 2: Evol-Instruct-style harder variants
../../../.venv/bin/python phase2_synthetic/judge.py        # LLM-as-judge: keep only faithful candidates
../../../.venv/bin/python phase2_synthetic/filter.py       # dedup + decontaminate vs gold + assemble
```

What each step is for:

- `paraphrase.py` (-> `data/gen_paraphrase.jsonl`): the simplest, safest expansion.
  Each seed query is reworded several ways that keep the same request, so the
  intent label is unchanged. This is the grounding rule: vary phrasing, never the
  verified label. `N_PER_SEED` controls how many per seed.
- `evolve.py` (-> `data/gen_evol.jsonl`): Evol-Instruct-style. Instead of just
  rewording, it evolves each seed into a harder, more realistic query along a few
  operators (`deepen`, `constraint`, `concretize`), still keeping the same intent.
  This teaches the classifier to hold the label steady under tougher input.
- `judge.py` (-> `data/judged.jsonl`): the LLM-as-judge gate. It asks the local
  model, per candidate, whether the message truly expresses its assigned intent,
  and keeps only faithful candidates scoring >= `KEEP_SCORE`. The header documents
  the judge's biases (length bias, and self-preference because the generator and
  judge are the same model; a stronger judge model is the real fix).
- `filter.py` (-> `data/train_synth.jsonl`): the mechanical quality gate.
  Decontaminates against the gold set (exact + char-shingle near-duplicate, the
  step that protects the evaluation), dedups against seeds/gold/each other, prints
  a per-intent balance report, and assembles the final file as all seeds plus the
  kept synthetic. Keeping the real seeds in the mix guards against mode collapse.

Output `data/train_synth.jsonl` is in the same messages format as the seed, ready
for Phase 3. Cost note: generation is one local call per seed (paraphrase) and one
per seed per operator (evolve); judging is one call per candidate, so the judge
step is the slow one. Lower `N_PER_SEED`, the `OPERATORS` set, or judge fewer rows
if a full pass is too slow on your hardware.

## Next

Phase 3 (a `phase3_train/` folder) trains the LoRA on `data/train_synth.jsonl`,
then re-runs `eval/evaluate.py --name seed-synth --adapter <adapter-dir>` to get
the real before/after on both axes against the recorded base baseline. The planned
comparison is three conditions: base vs seed-only fine-tune vs seed-plus-synthetic
fine-tune.
