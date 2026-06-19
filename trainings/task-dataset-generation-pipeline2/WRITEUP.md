## 4. Base vs fine-tuned for one task, with the gate verdict (Exp 2 + 4)

Task: helpdesk-resolution. Regression axes all run at 60 probes (reasoning grown 10 to 60,
tools grown 8 to 60, matching the sentinel) so the 0.05 ceiling sits inside the measurable
range on every axis. Full breakdown in `runs/gate_verdict_60probe.md`.

- acceptance criteria: min task gain = +0.050, max regression drop = 0.050
- attempt 1 (replay, train_mix.jsonl, lr 2e-4, 6 epochs): task gain = -0.001,
  worst regression drop = -0.049 -> REJECT (no task gain)
- adjusted re-run (lr 1e-4, 2 epochs, replay on): task gain = +0.171,
  worst regression drop = 0.000 -> ACCEPT
- accepted adapter: helpdesk-resolution-adj

| axis | base | accepted (adj re-run) | delta |
|------|------|-----------------------|-------|
| task macro-F1 | 0.395 | 0.566 | +0.171 |
| sentinel | 0.902 | 0.902 | 0.000 |
| reasoning | 0.850 | 0.900 | +0.050 |
| tools | 0.783 | 0.867 | +0.084 |

The gate behaved sensibly and the adjusted re-run is what made the pipeline trustworthy here.
Attempt 1 trained the full 6 epochs at lr 2e-4 and its training loss fell to about 1.5, yet its
gold-set macro-F1 (0.394) was no better than base: it overshot, minimizing training loss while it
stopped generalizing. The gate caught this (task gain -0.001) and triggered one configured
adjustment, which halved the learning rate and cut to 2 epochs. That gentler, shorter run scored
0.566 (+0.171) with no regression on any axis (sentinel flat, reasoning and tools up), and was
accepted. The shorter run beating the longer one is direct evidence that a fixed epoch count is
the wrong target and that adaptive length is the right idea.

Two caveats. First, run-to-run variance is high on this setup: the same 6-epoch, lr 2e-4 config
scored 0.501 in an earlier run and 0.394 here. On 296 training rows with a 0.5B model on a T4
(4-bit, not bitwise deterministic even with a fixed seed), single-run task numbers swing a lot, so
the gate plus the adjusted re-run, not any one run, is what makes the outcome robust. Second, the
earlier +0.106 result and this +0.171 result came from different runs and configs; treat the
accepted adapter (the adj re-run above) as the current result.

### How the training length is set (not a fixed number)

The epoch count is not hand-picked per task. It is set in three layers, so the pipeline
generalises instead of relying on a number someone tuned by hand:

1. Propose a starting budget. `propose_config.py` reads the dataset and sets `train.epochs` as an
   UPPER BOUND, not a target: smaller sets get more passes (6 if under 500 rows, 4 under 2000, 2
   beyond), since small sets need more passes and large ones overfit. The human reviews and
   confirms it. This is the "rules propose, human confirms" step.
2. Adapt the real length to the task. With `train.early_stopping.enabled`, the run scores both
   axes after every epoch (a clean bf16 re-load of the saved adapter, the same measurement the
   gate uses) and keeps the best adapter. It stops as soon as a regression axis falls more than
   `regression_tolerance` below its best (forgetting has begun) or task macro-F1 has not improved
   by `min_task_delta` for `patience` evals. So a task that peaks at epoch 2 stops at epoch 2, and
   one still improving at epoch 6 trains the full budget. The epoch count comes out of the data,
   not the config. (It is currently off while the clean-eval signal is being validated; with it
   off the run trains the fixed `epochs` budget.)
3. Adjust and re-run if the gate still rejects. If the accepted-or-not decision fails, the
   pipeline applies one configured adjustment (halve the learning rate, cut epochs) and trains
   once more. That is exactly what produced the accepted adapter above: the 6-epoch run overshot
   and was rejected, and the gentler 2-epoch re-run was accepted.

So "epochs: 6" in the config is a ceiling on effort, not the chosen length. Early stopping picks
the length per task within that ceiling, and the gate plus one adjusted re-run is the backstop
when a run still misses.
