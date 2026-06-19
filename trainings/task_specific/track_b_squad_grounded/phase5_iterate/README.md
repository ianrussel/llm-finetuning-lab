# Phase 5: iterate on the dominant failure mode

A grounded-QA model fails in three distinct ways, and each needs different data,
so iteration here is about naming which one dominates and generating exactly that.

```
../../../.venv/bin/python phase5_iterate/error_analysis.py   # name the failure mode from preds
../../../.venv/bin/python phase5_iterate/targeted.py         # generate more of that mode (Ollama up)
../../../.venv/bin/python phase5_iterate/build_v2.py         # judge + decontaminate + RAFT assemble
../../../.venv/bin/python phase3_train/train.py --data data/train_synth_v2.jsonl --name seed-synth-v2
../../../.venv/bin/python eval/evaluate.py --name seed-synth-v2 --adapter phase3_train/lora-seed-synth-v2
../../../.venv/bin/python eval/compare.py --names base seed seed-synth seed-synth-v2 \
    --effect-pair seed-synth seed-synth-v2
```

## The three failure modes

- `error_analysis.py` (-> `data/error_analysis.json`) reads `preds_seed-synth.jsonl`
  and splits the misses into:
  - **hallucination**: an unanswerable question got an answer instead of an
    abstention. Fix: more, and harder, unanswerable examples.
  - **over-abstention**: an answerable question got `not in the context` when the
    answer was present. Fix: more answerable examples.
  - **wrong answer**: an answerable question was answered with low token-F1. Fix:
    more answerable comprehension examples.

  Whichever mode clears its bar (`HALLUCINATION_BAR`, `ANSWERABLE_BAR`) becomes a
  target. If none does, the model is balanced and the script says to stop.

- `targeted.py` (-> `data/gen_targeted.jsonl`) reuses the Phase 2 generators but
  weighted toward the targeted mode(s), from fresh passages.
- `build_v2.py` (-> `data/train_synth_v2.jsonl`) runs the targeted candidates
  through the same Phase 2 discipline (faithfulness judge, gold decontamination,
  dedup against `train_synth.jsonl`, RAFT assembly) and appends them.

## Reading the result and stopping

`compare.py --effect-pair seed-synth seed-synth-v2` shows the move on every
sub-metric. For grounded QA the usual story is a trade: more unanswerable data
lifts abstention accuracy and cuts hallucination, but can nudge up over-abstention
(the model gets shy and abstains on answerable questions too). Watch grounded_score
and the regression sentinel together. Stop when grounded_score clears the bar and
the sentinel holds, or when another round stops moving the number. Record the
decision and the reason in `PROGRESS.md`.
