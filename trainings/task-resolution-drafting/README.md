# Open-ended task, scored by an LLM judge

The first two tasks are classification scored by macro-F1. A general module also needs open-ended
tasks, where there is no single right answer, like drafting a support reply. This folder adds that
shape: same config-driven pipeline, multi-seed median, and regression axis, but the **task axis is
an LLM-judge win-rate** instead of macro-F1.

- **Task:** draft the agent reply to a Tobi-Bueck support ticket (the `answer` field). Open-ended.
- **Judge:** a stronger model (Qwen2.5-7B-Instruct, 4-bit) compares the base model's answer and the
  fine-tuned model's answer to the same ticket and says which is better. The fraction the fine-tuned
  model wins (ties count half) is the **win-rate**, which the gate uses in place of task macro-F1.
- **Bias control:** every pair is judged in both A/B orders, so a judge that just prefers the first
  answer cancels out.
- **Calibrate first:** `calibrate.py` checks the judge against your own human judgments before you
  trust it to gate.

## What is reused vs new

Reused unchanged from the classification pipeline (the model-agnostic core): `common_p2.py`,
`train_from_config.py`, `evaluate_from_config.py` (its load/gen/probe helpers). New here:

```
judge.py             pairwise LLM judge -> win-rate (both-orders debias)
evaluate_gen.py      generate each model's answers on held-out tickets + score regression probes
gate_gen.py          accept iff judge win-rate >= min_win_rate AND no probe regressed
pipeline_gen.py      orchestrate: base -> train -> answers -> judge -> gate, multi-seed median
prepare_data_gen.py  Tobi-Bueck -> (ticket -> reference reply) dataset
calibrate.py         judge vs human agreement, run before trusting the judge
config.yaml          base_model, judge.model, generative acceptance (min_win_rate)
```

## Run order

```
pip install datasets
../../.venv/bin/python prepare_data_gen.py        # build data/ from Tobi-Bueck
#   review data/gold.jsonl, set confirmed: true in config.yaml

../../.venv/bin/python pipeline_gen.py            # base -> train -> answers -> judge -> gate

# calibrate the judge (do this before trusting a verdict): hand-label data/calibration.jsonl,
# pairing base vs candidate answers from runs/result_*.json, then:
../../.venv/bin/python calibrate.py
```

## How the gate reads here

- **task axis:** judge win-rate of candidate over base, accept at `min_win_rate` (default 0.55, i.e.
  the judge prefers the fine-tuned model on a clear majority of tickets).
- **regression axis:** unchanged, sentinel / reasoning / tool probes must not drop more than
  `max_regression_drop` vs base.
- **multi-seed:** each seed's candidate is judged vs the same base answers; the gate decides on the
  median win-rate across seeds and ships the best passing seed.

## Notes / caveats

- **Calibrate before trusting.** A win-rate from an uncalibrated judge is not evidence. Aim for
  about 0.8 agreement with your own decisive judgments first (`calibrate.py`).
- **Memory / time.** The judge is a 7B; it loads in 4-bit after the candidate answers are generated
  and the trained model is freed, so only one large model is resident at a time. Still heavy on a
  free T4. Use the 0.5B base for cheap iteration; swap up later.
- **Adaptive length is off for v1.** The clean-eval early stop scores macro-F1, which does not apply
  to generation. A ROUGE-vs-reference proxy early stop is the planned addition; for now the run
  trains the fixed `epochs` budget and the judge gates the result.
- **Model-agnostic, same as the other tasks:** retarget the trained model or the judge by changing
  `base_model` / `judge.model` in the config, no code edits.
