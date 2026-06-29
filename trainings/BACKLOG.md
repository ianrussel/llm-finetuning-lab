# Backlog

Priority order. The point of this file is so deferred ideas are tracked without pulling focus from
the current main task.

## Now (main focus)

- **Generative task with the LLM judge** (`task-resolution-drafting/`). Built: judge, win-rate gate,
  multi-seed loop, calibration, notebook. What's left is a *run*, not a build:
  1. Calibrate the judge (`calibrate.py` / notebook calibration cell) to about 0.8 agreement with
     your own decisive judgments. A win-rate is not trustworthy until this passes.
  2. Then run it (`pipeline_gen.py` / the notebook) and record the win-rate verdict.
  This is the one to push on.

## Deferred (nice-to-have; slot in when convenient)

- **eval-loss early-stop mode for big-model adaptive length.** Why deferred: the fixed-epoch + gate +
  multi-seed fallback is reliable (it accepted 2/2 on Phi-4-3.8B), so this is a nice-to-have, not a
  blocker. The two generate-based modes both fail on a 3.8B on a free T4: `eval_mode: clean` OOMs
  (second model copy), `eval_mode: resident` throws a batched-`generate()` shape error on the live
  QLoRA model. Fix: add `early_stopping.eval_mode: "loss"` that uses held-out eval **loss** (a forward
  pass only, no second copy, no `generate()`) as the task-plateau signal — which is how early stopping
  is normally done. Regression stays a gate check (or is skipped in-loop). Validate cheaply on the
  small Qwen before trusting it on a slow 3.8B run. See `task-support-tickets/MODEL_NOTES.md` #6.

- **Generative adaptive length** (`task-resolution-drafting/`): a ROUGE-vs-reference in-loop signal so
  the generative task can early-stop too (it trains fixed epochs today; macro-F1 early stop does not
  apply to open-ended output).

- **Tidy the human-in-the-loop loop**: the seeds -> confirm config -> approve a few samples -> go
  autonomous flow exists in pieces; make it one clean loop, especially the sample-approval checkpoint
  for the generative pipeline.

## Done (for context)

- Model-agnosticism across three families: Qwen 0.5B (pipeline2), SmolLM3-3B (task1), Phi-4-mini-3.8B
  (task2), same pipeline, config-only differences.
- Two-axis gate (task + regression) at 60 probes; multi-seed median; clean-eval early stopping with
  adaptive length on small models; sentinel probe budget fixed for reasoning models.
- Knowledge-absorption third axis (`task-support-tickets/task2/{build_knowledge,knowledge}.py`).
- Generative + LLM-judge pipeline and a second/third corpus (Tobi-Bueck support tickets, public Jira).
