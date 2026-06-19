# Phase 5: iterate on the weak spots

One training pass is rarely the end. Phase 4 says how good the seed+synthetic
model is and, through `compare.py`, roughly where it is weak. Phase 5 closes the
loop: read the gold misses, name the pattern, generate data aimed at exactly that
weakness, retrain, and re-measure on the same two axes. The point is targeted data,
not just more data.

## The loop

Three new scripts do the analysis and the targeted generation; the retrain and
re-measure reuse Phase 3 and Phase 4 unchanged.

```
# 1. name the pattern from the current best model's gold misses
../../../.venv/bin/python phase5_iterate/error_analysis.py

# 2. generate contrastive data for the named confusable pairs (Ollama up)
../../../.venv/bin/python phase5_iterate/targeted.py

# 3. judge + decontaminate + assemble the next training set
../../../.venv/bin/python phase5_iterate/build_v2.py

# 4. retrain on the extended set
../../../.venv/bin/python phase3_train/train.py --data data/train_synth_v2.jsonl --name seed-synth-v2

# 5. re-measure both axes and line up the iteration delta
../../../.venv/bin/python eval/evaluate.py --name seed-synth-v2 --adapter phase3_train/lora-seed-synth-v2
../../../.venv/bin/python eval/compare.py --names base seed seed-synth seed-synth-v2 \
    --effect-pair seed-synth seed-synth-v2
```

## What each new script does

- `error_analysis.py` (-> `data/error_analysis.json`): reads the per-row
  predictions Phase 4 wrote (`data/preds_seed-synth.jsonl`) and turns the misses
  into a plan. It ranks intents by F1 weakest-first, splits weakness into recall
  (the model misses the intent) vs precision (it over-fires it on neighbours),
  and for each weak intent finds its top confuser, the single wrong label it most
  often collapses into. A classifier fails as confusable pairs, so the `targets`
  list it writes is `{intent, confuser, n_missed}`, ready for the next step.
- `targeted.py` (-> `data/gen_targeted.jsonl`): the iteration-specific generator.
  For each target it takes real seed messages of the weak intent and asks the local
  model to rewrite them so the request is unambiguously that intent and could not
  be read as its confuser. This is contrastive, not undirected: it attacks the
  exact pair `error_analysis.py` named. The grounding rule is unchanged from
  Phase 2, the verified label never moves, only the phrasing is sharpened.
- `build_v2.py` (-> `data/train_synth_v2.jsonl`): the same quality gate as Phase 2,
  no shortcut for being on purpose. It judges each candidate with the Phase 2
  LLM-as-judge, decontaminates against the sacred gold set (exact + near-dup),
  dedups against everything already in `train_synth.jsonl` and against itself, then
  assembles `train_synth_v2.jsonl = train_synth.jsonl + kept targeted rows`. The v1
  set already carries the real seeds, so the mode-collapse guard rides along.

## Reading the result

`compare.py --effect-pair seed-synth seed-synth-v2` lines up all four conditions
on both axes and breaks the iteration move down per intent, so the question is
sharp: did targeting the confusable pairs lift their F1, and did it cost anything
elsewhere or on the regression sentinel. Two cautions when reading it:

- Per-intent F1 sits on 20 gold rows each, so it moves in coarse steps. Trust the
  direction and rough size, not the third decimal.
- Targeting a pair can rob its neighbour (a label becomes an attractor). That is
  exactly why the per-intent breakdown and the sentinel both stay in view.

## When to stop

Stop when the headline metric clears the bar and the regression stays in
tolerance, or when another round stops moving the number. Each iteration is one
turn of: error-analyze, target, retrain, re-measure. Record the decision and the
reason in `PROGRESS.md` so the stopping point is deliberate, not just where the
session ended.
