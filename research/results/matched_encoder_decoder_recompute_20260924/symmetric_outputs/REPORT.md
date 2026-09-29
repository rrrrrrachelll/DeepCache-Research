# Symmetric Matched Encoder–Decoder Path Recompute

## Protocol and integrity

- SD1.5, DPM-Solver++ 20 steps, guidance scale 7.5.
- DeepCache balanced8-A refresh steps: `[0, 2, 5, 7, 10, 13, 16, 19]`.
- Probe steps: 4 and 9 (cache age 2).
- Data: 10 analysis prompts x seeds 101 and 202; consumed held-out prompts excluded.
- Each partial path symmetrically recomputes shallow encoder blocks and their matching decoder blocks while retaining the cached deeper core. `matched_0123` caches only the mid block; `full_unet` is exact.
- 20/20 traces, 400 diffusion steps, 200 path probes.
- Full-UNet control maximum guided-noise error: `0`.

## Overall results

| Path | Modules | Guided error | Recovery | Positive prompts | Incremental time | Saved vs full |
|---|---:|---:|---:|---:|---:|---:|
| matched_0 | 13 | 0.109282 | -0.94% | 2/10 | 1634.1 ms | 766.2 ms |
| matched_01 | 27 | 0.106221 | 1.10% | 4/10 | 2040.1 ms | 360.2 ms |
| matched_012 | 41 | 0.062821 | 41.29% | 10/10 | 2296.4 ms | 104.0 ms |
| matched_0123 | 49 | 0.057598 | 45.75% | 10/10 | 2365.2 ms | 35.1 ms |
| full_unet | 50 | 0.000000 | 100.00% | 10/10 | 2400.3 ms | 0.0 ms |

## Results by diffusion step

| Path | Step 4 recovery | Step 9 recovery | Step 4 positive | Step 9 positive |
|---|---:|---:|---:|---:|
| matched_0 | -1.38% | -0.51% | 8/20 | 9/20 |
| matched_01 | -6.57% | 8.76% | 4/20 | 14/20 |
| matched_012 | 25.92% | 56.65% | 16/20 | 20/20 |
| matched_0123 | 28.61% | 62.90% | 16/20 | 20/20 |
| full_unet | 100.00% | 100.00% | 20/20 | 20/20 |

## Decision

Partial paths passing the predefined gate: `['matched_012', 'matched_0123']`.

The best partial path by mean recovery was `matched_0123` at 45.75%. Symmetric recomputation tests whether refreshing matched shallow skip producers and consumers can recover guided-noise error while retaining the cached deep core. The exact full-UNet control confirms the upper bound and instrumentation correctness.

A partial path is eligible for closed-loop image evaluation only if it reaches 20% mean recovery, is nonnegative at both probe steps, and is positive for at least 7/10 prompt groups. Its isolated time saving versus `full_unet` must also be large enough to justify the added implementation complexity.

Timing values are isolated probe costs and must not be interpreted as end-to-end generation latency. They are suitable for comparing frontiers within this experiment.
