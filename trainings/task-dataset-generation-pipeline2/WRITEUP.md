## 4. Base vs fine-tuned for one task, with the gate verdict (Exp 2 + 4)

Task: helpdesk-resolution. Regression axes all run at 60 probes (reasoning grown 10 to 60,
tools grown 8 to 60, matching the sentinel) so the 0.05 ceiling sits inside the measurable
range on every axis. Each stage trains a couple of seeds and the gate decides on the MEDIAN
across them. Full breakdown in `runs/gate_verdict_60probe.md`.

- acceptance criteria: min task gain = +0.050, max regression drop = 0.050
- attempt 1 (replay, train_mix.jsonl, lr 2e-4, 6 epochs), seeds [0, 1]: median task gain = +0.036,
  1/2 seeds passed -> REJECT
- adjusted re-run (lr 1e-4, 2 epochs, replay on), seeds [0, 1]: median task gain = +0.112,
  2/2 seeds passed -> ACCEPT
- accepted adapter: helpdesk-resolution-adj-s0 (best of the passing seeds)

| axis | base | accepted (adj-s0) | delta |
|------|------|-------------------|-------|
| task macro-F1 | 0.395 | 0.533 | +0.138 |
| sentinel | 0.902 | 0.951 | +0.049 |
| reasoning | 0.850 | 0.917 | +0.067 |
| tools | 0.783 | 0.900 | +0.117 |

The run shows both the gate and the multi-seed guard doing their jobs. Attempt 1 (6 epochs, lr
2e-4) scored +0.081 on seed 0 but -0.009 on seed 1: a single-seed run would have shipped the
lucky seed and called it a win, but the median (+0.036) sat below the floor, so it was correctly
rejected. The pipeline then applied its one configured adjustment (halve the learning rate, cut
to 2 epochs). That gentler, shorter run passed BOTH seeds (+0.138 and +0.087, median +0.112) with
no regression on any axis, and was accepted; the best of the two adapters was kept.

### How the training length is set (not a fixed number)

The epoch count is not hand-picked per task. It is set in three layers, so the pipeline
generalises instead of relying on a number someone tuned by hand:

1. Propose a starting budget. `propose_config.py` reads the dataset and sets `train.epochs` as an
   UPPER BOUND, not a target: smaller sets get more passes (6 if under 500 rows, 4 under 2000, 2
   beyond), since small sets need more passes and large ones overfit. The human reviews and
   confirms it.
2. Adapt the real length to the task. With `train.early_stopping.enabled`, the run scores both
   axes after every epoch (a clean bf16 re-load of the saved adapter, the same measurement the
   gate uses) and keeps the best adapter. It stops as soon as a regression axis falls more than
   `regression_tolerance` below its best (forgetting has begun) or task macro-F1 has not improved
   by `min_task_delta` for `patience` evals. (It is currently off while the clean-eval signal is
   being validated; with it off the run trains the fixed `epochs` budget.)
3. Adjust and re-run if the gate still rejects. If the median verdict fails, the pipeline halves
   the learning rate and cuts epochs and trains once more. That is what produced the accepted
   adapter above.

So "epochs: 6" is a ceiling on effort, not the chosen length.

### Robust to run-to-run noise: median across seeds

Because the same config swung 0.501 vs 0.394 across earlier runs, the accept/reject must not ride
on a single noisy run. Each stage trains a couple of seeds (`train.seeds`, default `[0, 1]`) and
the gate decides on the MEDIAN task gain and MEDIAN worst regression drop, then ships the best
adapter among the seeds that individually pass. This run is the proof: attempt 1's seeds were
+0.081 and -0.009, straddling the floor, and the median correctly rejected; the adjusted re-run's
seeds were +0.138 and +0.087, both clearly passing, and the median accepted. The longer 6-epoch
schedule is not only lower on average, it is higher variance (range [-0.009, +0.081]) than the
gentle 2-epoch one (range [+0.087, +0.138]), which is exactly why a fixed, single-run epoch count
is risky. The cost is one extra training run per stage; set `seeds: [0]` to disable.
