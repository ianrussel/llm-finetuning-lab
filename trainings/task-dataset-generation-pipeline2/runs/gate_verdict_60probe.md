# Gate verdict: 60-probe, multi-seed run

Final accepted adapter: `helpdesk-resolution-adj-s0` (adjusted re-run: train_mix.jsonl, lr 1e-4, 2 epochs, seed 0)
Base model: Qwen/Qwen2.5-0.5B-Instruct
Probe sets: sentinel, reasoning, and tools all at 60 probes (reasoning grown 10 to 60, tools grown 8 to 60, matching the sentinel) so the 0.05 ceiling sits inside the measurable range on every axis.
Decision rule: each stage trains seeds [0, 1]; the gate decides on the MEDIAN across seeds and ships the best adapter among those that individually pass.

## Verdict: ACCEPT

Accepted adapter `helpdesk-resolution-adj-s0`: task gain +0.138, no regression (worst drop -0.049).

| Axis | Base | Accepted (adj-s0) | Delta |
|------|------|-------------------|-------|
| task macro-F1 | 0.395 | 0.533 | +0.138 |
| sentinel | 0.902 | 0.951 | +0.049 |
| reasoning | 0.850 | 0.917 | +0.067 |
| tools | 0.783 | 0.900 | +0.117 |

## Both stages, per seed

**Attempt 1 (replay, 6 epochs, lr 2e-4):**

| seed | macro-F1 | task gain | worst drop | verdict |
|------|----------|-----------|------------|---------|
| 0 | 0.476 | +0.081 | -0.033 | ACCEPT |
| 1 | 0.386 | -0.009 | -0.033 | REJECT |
| median | — | +0.036 | -0.033 | REJECT (1/2 passed) |

**Adjusted re-run (2 epochs, lr 1e-4):**

| seed | macro-F1 | task gain | worst drop | verdict |
|------|----------|-----------|------------|---------|
| 0 | 0.533 | +0.138 | -0.049 | ACCEPT |
| 1 | 0.482 | +0.087 | 0.000 | ACCEPT |
| median | — | +0.112 | -0.025 | ACCEPT (2/2 passed) |

## What the run demonstrates

- **The multi-seed guard caught real noise.** Attempt 1 scored +0.081 on seed 0 and -0.009 on seed 1. A single-seed run would have shipped seed 0 and reported a +0.081 win on luck. The median (+0.036) sat below the 0.05 floor, so it was correctly rejected.
- **The accepted run passed on agreement, not luck.** The adjusted re-run cleared the gate on both seeds (+0.138 and +0.087), so the median accept is robust.
- **Longer is not just lower, it is noisier.** The 6-epoch, lr 2e-4 schedule has gain range [-0.009, +0.081], straddling the floor; the gentle 2-epoch, lr 1e-4 schedule has range [+0.087, +0.138], both clearly passing. That higher variance is exactly why a fixed, single-run epoch count is risky and why the median-across-seeds rule matters.
- **No regression anywhere.** Every axis on the accepted adapter is up; the reported worst drops are negative.

## Resolution reference

| Axis | Probes | Resolution (1/n) | One flip reads as |
|------|--------|------------------|-------------------|
| sentinel | 60 | 0.017 | noise (tolerated) |
| reasoning | 60 | 0.017 | noise (tolerated) |
| tools | 60 | 0.017 | noise (tolerated) |

With the 0.05 ceiling, it takes 4 or more probe flips on a single axis to trip a reject, so the threshold sits inside the measurable range on every axis.
