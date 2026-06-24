## 4. Base vs fine-tuned for one task, with the gate verdict (Exp 2 + 4)

Task: helpdesk-resolution. Regression axes all run at 60 probes (reasoning grown 10 to 60,
tools grown 8 to 60, matching the sentinel) so the 0.05 ceiling sits inside the measurable
range on every axis. Each stage trains a couple of seeds, scores both axes after every epoch
(clean-eval early stopping), and the gate decides on the MEDIAN across seeds. Full breakdown in
`runs/gate_verdict_60probe.md`.

- acceptance criteria: min task gain = +0.050, max regression drop = 0.050
- attempt 1 (replay, train_mix.jsonl, lr 2e-4, 6 epochs), seeds [0, 1]: median task gain = +0.040,
  1/2 seeds passed -> REJECT
- adjusted re-run (lr 1e-4, 2 epochs, replay on), seeds [0, 1]: median task gain = +0.109,
  2/2 seeds passed -> ACCEPT
- accepted adapter: helpdesk-resolution-adj-s0 (best of the passing seeds)

| axis | base | accepted (adj-s0) | delta |
|------|------|-------------------|-------|
| task macro-F1 | 0.395 | 0.549 | +0.154 |
| sentinel | 0.902 | 0.852 | -0.050 (at the edge) |
| reasoning | 0.850 | 0.883 | +0.033 |
| tools | 0.783 | 0.883 | +0.100 |

This run exercised the whole stack together and each piece behaved. Adaptive length kept different
best epochs per seed and stopped early. Patience 2 let a dip ride: attempt-1 s0 task went 0.436,
0.452, 0.423, 0.411 and it stopped only after two flat evals, keeping the peak at epoch 2 rather
than bailing on the first dip. The median across seeds rejected attempt 1 (median +0.040, only 1/2
seeds passed) and accepted the gentler re-run (median +0.109, 2/2). The accepted adapter has a
large task gain (+0.154); honestly, its sentinel dropped to the 0.05 edge (worst drop 0.049, just
inside tolerance), while the other passing seed was cleaner on regression but lower on task.

### How the training length is set (not a fixed number)

1. Propose a starting budget. `propose_config.py` sets `train.epochs` as an UPPER BOUND from the
   dataset size (6 if under 500 rows, 4 under 2000, 2 beyond). The human reviews and confirms.
2. Adapt the real length. With early stopping on, the run scores both axes after every epoch (a
   clean bf16 re-load of the saved adapter, the same measurement the gate uses), keeps the best
   adapter, and stops when a regression axis falls more than `regression_tolerance` below its best
   or task macro-F1 has not improved by `min_task_delta` for `patience` (2) evals.
3. Adjust and re-run if the median still rejects. Halve the learning rate, cut epochs, train once
   more. That is what produced the accepted adapter above.

So "epochs: 6" is a ceiling on effort, not the chosen length.

### Robust to run-to-run noise: median across seeds

The same config swung 0.501 vs 0.394 across earlier runs, so the verdict cannot ride on one run.
Each stage trains seeds `[0, 1]` and the gate decides on the median task gain and median worst
drop, shipping the best adapter among the seeds that individually pass. This run shows it working:
attempt 1's seeds were +0.016 and +0.065, median +0.040, correctly rejected; the re-run's were
+0.154 and +0.065, median +0.109, accepted.

### The learning-rate finding

Across three multi-seed runs the 6-epoch / lr 2e-4 attempt failed the median while the lr 1e-4
adjusted re-run passed 2/2. Early stopping already trims the hot run to about two effective epochs,
so the difference is the learning rate: 2e-4 overshoots and is high variance, 1e-4 generalises and
is consistent. The default `learning_rate` is now 1e-4 so attempt 1 uses the winning rate directly
instead of always falling through to the re-run, which also saves about half the compute.
