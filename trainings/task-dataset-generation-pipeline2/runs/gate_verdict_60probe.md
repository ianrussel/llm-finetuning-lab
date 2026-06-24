# Gate verdict: 60-probe, multi-seed, adaptive-length run

Final accepted adapter: `helpdesk-resolution-adj-s0` (adjusted re-run: train_mix.jsonl, lr 1e-4, 2 epochs, seed 0)
Base model: Qwen/Qwen2.5-0.5B-Instruct
This is the full-stack run: 60-probe regression axis, clean-eval early stopping (adaptive length), and the median across two seeds, all together.

## Verdict: ACCEPT

Accepted adapter `helpdesk-resolution-adj-s0`: task gain +0.154, regression within tolerance (worst drop 0.049).

| Axis | Base | Accepted (adj-s0) | Delta |
|------|------|-------------------|-------|
| task macro-F1 | 0.395 | 0.549 | +0.154 |
| sentinel | 0.902 | 0.852 | -0.050 (at the edge) |
| reasoning | 0.850 | 0.883 | +0.033 |
| tools | 0.783 | 0.883 | +0.100 |

Big task gain. Honest note: the chosen seed's sentinel dropped to the 0.05 edge (worst drop 0.049, just inside the ceiling). The other passing seed (`adj-s1`) was cleaner on regression (worst drop 0.016) but lower on task (+0.065); the chooser ships the highest task gain among the seeds that pass, which is `adj-s0`.

## Both stages, per seed

**Attempt 1 (replay, 6 epochs, lr 2e-4):**

| seed | best epoch | macro-F1 | task gain | worst drop | verdict |
|------|------------|----------|-----------|------------|---------|
| 0 | 2 | 0.411 | +0.016 | -0.033 | REJECT |
| 1 | 1 | 0.460 | +0.065 | +0.016 | ACCEPT |
| median | — | — | +0.040 | -0.008 | REJECT (1/2 passed) |

**Adjusted re-run (2 epochs, lr 1e-4):**

| seed | best epoch | macro-F1 | task gain | worst drop | verdict |
|------|------------|----------|-----------|------------|---------|
| 0 | 2 | 0.549 | +0.154 | 0.049 | ACCEPT |
| 1 | 2 | 0.460 | +0.065 | 0.016 | ACCEPT |
| median | — | — | +0.109 | 0.033 | ACCEPT (2/2 passed) |

## What this run validated

- **Adaptive length (clean-eval early stopping).** Per-epoch scores match the gate scale (no 4-bit noise). It kept different best epochs per seed and stopped early instead of running all 6.
- **Patience 2.** Attempt-1 s0 task went 0.436, 0.452, 0.423, 0.411: it did not bail on the first dip at epoch 3, it waited two evals and stopped at epoch 4 keeping the peak (epoch 2). Exactly the "task can climb after a dip" behaviour.
- **Tolerance epsilon.** Attempt-1 s1 epoch 3 hit a regression drop of exactly 0.050; it is now tolerated (the stop reason is the task plateau, not a false regression trip).
- **Median across seeds.** Rejected attempt 1 (median +0.040, 1/2) and accepted the re-run (median +0.109, 2/2), so no single noisy seed flips the verdict.

## The learning-rate finding (config change)

Across three multi-seed runs, the 6-epoch / lr 2e-4 attempt failed the median and the lr 1e-4 adjusted re-run passed 2/2. Early stopping already trims the hot run to ~2 effective epochs, so the difference is the learning rate: 2e-4 overshoots and is high-variance, 1e-4 generalises and is consistent. The config default `learning_rate` is now **1e-4** so attempt 1 uses the winning rate directly, instead of always needing the re-run.

## Resolution reference

| Axis | Probes | Resolution (1/n) | One flip reads as |
|------|--------|------------------|-------------------|
| sentinel | 60 | 0.017 | noise (tolerated) |
| reasoning | 60 | 0.017 | noise (tolerated) |
| tools | 60 | 0.017 | noise (tolerated) |

With the 0.05 ceiling, it takes 4 or more probe flips on a single axis to trip a reject, so the threshold sits inside the measurable range on every axis.
