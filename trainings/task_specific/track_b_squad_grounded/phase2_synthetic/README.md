# Phase 2: document-grounded synthetic data

Track A paraphrased and evolved existing labelled examples. Track B is different:
there is no label to vary, there is a corpus. So Phase 2 reads passages and
GENERATES grounded QA from them, the document-grounded approach behind tools like
Bonito and Augmentoolkit, rolled by hand here on a local Ollama model so nothing
leaves the machine.

Setup once:

```
ollama pull qwen2.5:3b-instruct
ollama serve     # if not already running
```

Then, in order:

```
../../../.venv/bin/python phase2_synthetic/qgen.py      # passages -> QA candidates
../../../.venv/bin/python phase2_synthetic/judge.py     # faithfulness gate
../../../.venv/bin/python phase2_synthetic/filter.py    # dedup + decontaminate + RAFT assemble
```

## What each step does

- `qgen.py` (-> `data/gen_qa.jsonl`): for each passage it generates two kinds of
  question. **Answerable** ones come with the shortest exact answer span, and a
  grounding guard drops any whose answer does not occur verbatim in the passage.
  **Unanswerable** ones are on-topic for the passage but not answered by it, and
  their target is the fixed abstention string. The unanswerable pass is the part
  that has no analogue in Track A: it manufactures the "I do not know" training
  signal. Knobs: `--answerable-per-passage`, `--unanswerable-per-passage`, `--limit`.
- `judge.py` (-> `data/judged.jsonl`): the LLM-as-judge gate, with a different
  question per candidate type. Answerable pairs are kept only if the judge calls
  the answer correct and supported (score >= `KEEP_SCORE`). Unanswerable questions
  are kept only if the judge confirms the passage genuinely cannot answer them,
  which is the check that stops the model being taught to abstain on answerable
  questions. Same honesty caveat as Track A: generator and judge are the same
  model, so a stronger judge model is the real upgrade.
- `filter.py` (-> `data/train_synth.jsonl`): the mechanical gate. Decontaminates
  against the gold set (question near-dup and gold-oracle passage), dedups against
  seeds and itself, then RAFT-assembles each kept candidate (oracle + distractors,
  shuffled) and writes the final set as all seeds plus kept synthetic. Keeping the
  real seeds in guards against mode collapse, and the answerable/unanswerable
  balance is printed so abstention does not get swamped.

Cost note: generation is a couple of calls per passage and judging is one call per
candidate, so judging is the slow step. Lower `--limit` or the per-passage counts
if a full pass is too slow.
