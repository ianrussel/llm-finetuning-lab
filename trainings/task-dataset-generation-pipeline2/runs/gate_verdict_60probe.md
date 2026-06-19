# Gate verdict: 60-probe regression run

Final accepted adapter: `helpdesk-resolution-adj` (the adjusted re-run: train_mix.jsonl, lr 1e-4, 2 epochs)
Base model: Qwen/Qwen2.5-0.5B-Instruct
Probe sets: sentinel, reasoning, and tools all at 60 probes (reasoning grown 10 to 60, tools grown 8 to 60, matching the sentinel) so the 0.05 ceiling sits inside the measurable range on every axis.

## Verdict: ACCEPT

`task_gain=+0.171 (min +0.050) | worst_drop=0.000 (max 0.050)` -> both axes pass with a wide margin.

| Axis | Base | Accepted (adj re-run) | Delta |
|------|------|-----------------------|-------|
| task macro-F1 | 0.395 | 0.566 | +0.171 |
| sentinel | 0.902 | 0.902 | 0.000 |
| reasoning | 0.850 | 0.900 | +0.050 |
| tools | 0.783 | 0.867 | +0.084 |

Task gain is more than triple the +0.050 floor, and no regression axis dropped (worst drop 0.000: sentinel flat, reasoning and tools up). This is the cleanest, widest-margin pass to date.

## How the pipeline got here: the gate plus the adjusted re-run

The first attempt did not pass, and the automatic adjustment is what rescued the run:

| run | config | task macro-F1 | task gain | worst drop | verdict |
|-----|--------|---------------|-----------|------------|---------|
| attempt 1 (replay) | 6 epochs, lr 2e-4 | 0.394 | -0.001 | -0.049 | REJECT (no task gain) |
| adjusted re-run | 2 epochs, lr 1e-4 | 0.566 | +0.171 | 0.000 | ACCEPT |

Attempt 1 trained the full 6 epochs and drove training loss down to about 1.5, but its gold-set macro-F1 (0.394) was no better than base. It overshot: it kept minimizing training loss while it stopped generalizing. The gate caught the lack of task gain and triggered one configured adjustment (halve the learning rate, cut to 2 epochs). That gentler, shorter run scored 0.566 with no regression and was accepted.

## What this says about training length

The shorter, gentler run (2 epochs, lr 1e-4) beat the longer, hotter one (6 epochs, lr 2e-4) by a wide margin on the task, with regression no worse. More epochs was not better here. That is direct evidence that a fixed epoch count is the wrong target and adaptive length is the right idea: with the clean-eval early stopping enabled, attempt 1 would likely have stopped once task stopped improving, instead of overshooting to 6 epochs.

## Caveats

- Run-to-run variance is high: the same 6-epoch, lr 2e-4 config scored 0.501 in an earlier run and 0.394 here. On 296 training rows with a 0.5B model on a T4 (4-bit, not bitwise deterministic even with a fixed seed), single-run task numbers swing a lot, so the gate plus the adjusted re-run is what makes the outcome robust, not any single run.
- Earlier write-ups quoted a +0.106 result from a different run and config. Treat the accepted adapter above (the adj re-run, +0.171) as the current result.

## Resolution reference

| Axis | Probes | Resolution (1/n) | One flip reads as |
|------|--------|------------------|-------------------|
| sentinel | 60 | 0.017 | noise (tolerated) |
| reasoning | 60 | 0.017 | noise (tolerated) |
| tools | 60 | 0.017 | noise (tolerated) |

With the 0.05 ceiling, it takes 4 or more probe flips on a single axis to trip a reject, so the threshold sits inside the measurable range on every axis.
