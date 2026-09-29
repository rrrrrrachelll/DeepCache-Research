# Matched Encoder–Decoder Path Recompute

## Protocol and integrity

- SD1.5, DPM-Solver++ 20 steps, guidance scale 7.5.
- DeepCache balanced8-A refresh steps: `[0, 2, 5, 7, 10, 13, 16, 19]`.
- Probe steps: 4 and 9 (cache age 2).
- Data: 10 analysis prompts x seeds 101 and 202; consumed held-out prompts excluded.
- Each path retains the cached shallower boundary, then recomputes the selected encoder block, all deeper encoder blocks, the mid block, and the full decoder.
- 20/20 traces, 400 diffusion steps, 200 path probes.
- Full-UNet control maximum guided-noise error: `0`.

## Overall results

| Path | Modules | Guided error | Recovery | Positive prompts | Incremental time |
|---|---:|---:|---:|---:|---:|
| decoder_only | 29 | 0.110674 | -3.23% | 1/10 | 1339.9 ms |
| down3_pair | 32 | 0.110752 | -3.24% | 1/10 | 1361.9 ms |
| down2_pair | 38 | 0.108683 | -1.98% | 3/10 | 1440.4 ms |
| down1_pair | 44 | 0.110027 | -5.18% | 1/10 | 1579.5 ms |
| full_unet | 50 | 0.000000 | 100.00% | 10/10 | 2411.9 ms |

## Results by diffusion step

| Path | Step 4 recovery | Step 9 recovery | Step 4 positive | Step 9 positive |
|---|---:|---:|---:|---:|
| decoder_only | -7.55% | 1.09% | 2/20 | 12/20 |
| down3_pair | -7.59% | 1.11% | 2/20 | 13/20 |
| down2_pair | -6.00% | 2.05% | 3/20 | 12/20 |
| down1_pair | -9.52% | -0.85% | 4/20 | 12/20 |
| full_unet | 100.00% | 100.00% | 20/20 | 20/20 |

## Decision

Partial paths passing the predefined gate: `[]`.

The best partial path by mean recovery was `down2_pair` at -1.98%. No partial matched path passed the recovery gate. Any retained shallow encoder boundary leaves stale hidden and skip states that conflict with the newly recomputed deeper path. The exact full-UNet control confirms that the instrumentation can recover the teacher output.

The experiment decides whether matched partial recomputation is viable under the predefined 20% recovery and prompt-stability gate. If no partial path passes, the remaining coherent option is a real full refresh at selected critical steps, followed by closed-loop image-quality and end-to-end latency evaluation.

Timing values are isolated probe costs and must not be interpreted as end-to-end generation latency. They are suitable for comparing frontiers within this experiment.
