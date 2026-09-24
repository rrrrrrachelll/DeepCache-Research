# Decoder Recompute Frontier Sweep

## Protocol and integrity

- SD1.5, DPM-Solver++ 20 steps, guidance scale 7.5.
- DeepCache balanced8-A refresh steps: `[0, 2, 5, 7, 10, 13, 16, 19]`.
- Probe steps: 4 and 9 (cache age 2).
- Data: 10 analysis prompts x seeds 101 and 202; consumed held-out prompts excluded.
- Each partial frontier uses cached upstream state and recomputes the selected real network block plus every downstream wrapped module.
- 20/20 traces, 400 diffusion steps, 240 frontier probes.
- Full-UNet control maximum guided-noise error: `0`.

## Overall results

| Frontier | Modules | Guided error | Recovery | Positive prompts | Incremental time |
|---|---:|---:|---:|---:|---:|
| up_block_0 | 7 | 0.108219 | -0.93% | 1/10 | 849.4 ms |
| up_block_1 | 15 | 0.111107 | -3.88% | 1/10 | 1105.3 ms |
| up_block_2 | 23 | 0.111478 | -4.10% | 1/10 | 1263.9 ms |
| up_block_3 | 28 | 0.111033 | -3.66% | 1/10 | 1313.0 ms |
| mid_decoder | 29 | 0.110674 | -3.23% | 1/10 | 1336.8 ms |
| full_unet | 50 | 0.000000 | 100.00% | 10/10 | 2405.2 ms |

## Results by diffusion step

| Frontier | Step 4 recovery | Step 9 recovery | Step 4 positive | Step 9 positive |
|---|---:|---:|---:|---:|
| up_block_0 | -1.53% | -0.32% | 2/20 | 8/20 |
| up_block_1 | -7.41% | -0.35% | 1/20 | 10/20 |
| up_block_2 | -8.63% | 0.42% | 2/20 | 11/20 |
| up_block_3 | -8.05% | 0.74% | 2/20 | 11/20 |
| mid_decoder | -7.55% | 1.09% | 2/20 | 12/20 |
| full_unet | 100.00% | 100.00% | 20/20 | 20/20 |

## Decision

Partial frontiers passing the predefined gate: `[]`.

The best partial frontier by mean recovery was `up_block_0` at -0.93%. No partial decoder suffix recovered the cached guided-noise error. Recomputing a fresh decoder from stale cached encoder/mid and skip features creates an inconsistent hybrid state; extending the frontier through the decoder does not fix that mismatch. The exact full-UNet control confirms that the instrumentation can recover the teacher output.

This experiment therefore rejects decoder-only suffix recomputation. The next local-recompute experiment should preserve skip consistency by recomputing matched encoder-to-decoder paths, beginning with the down-block that supplies the corresponding decoder skip plus the downstream mid/decoder path. A cheaper practical alternative is an additional full refresh at the critical early step, followed by closed-loop image and latency evaluation.

Timing values are isolated probe costs and must not be interpreted as end-to-end generation latency. They are suitable for comparing frontiers within this experiment.
