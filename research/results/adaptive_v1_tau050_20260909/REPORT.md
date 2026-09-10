# Adaptive DeepCache V1: Threshold 0.50

Implemented a standalone helper in `research/adaptive_cache.py`, leaving the
original helper and sampling pilot untouched. This run verifies the first
feature-aware/log-SNR policy on SD1.5 with deterministic second-order
DPM-Solver++ at 20 steps. It is an exploratory implementation, not a validated
improvement across prompts or an equal-cost benchmark.

## Reproduce

From the repository root, choose a new output directory:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/run_adaptive.py --output research/results/adaptive_new_run --risk-threshold 0.50
/root/miniconda3/envs/deepcache/bin/python -B -m unittest research.test_adaptive_cache research.test_risk_policy -v
```

The explicit 0.50 flag is required to reproduce this setting. The code's 0.65
default is the initial diagnostic setting, retained for reproducibility. Its
separate run is in `../adaptive_v1_20260909/`; all dynamic signals there were
masked by the maximum-age guard. The 0.50 threshold was chosen from the observed
seed-42 risk scale, so these cases are calibration/development cases, not a
held-out test set. No further threshold sweep was performed.

## Policy

Observe fresh `conv_in` features before any potentially cached down block. Pool
4x4 and reduce in float32. E is relative mean absolute change from the preceding
call. It is an input-feature-change proxy, not true deep-feature error or local
solver truncation error. It precedes time/text conditioning within the UNet.

Use lambda=log(alpha)-log(sigma), half log-SNR. Accumulate
`R += abs(delta_lambda) * (1 + 2 * E)` and refresh at `R >= 0.50`.
Reset R on refresh. Also refresh the first two calls, final call, at cache age
3 (at most two consecutive reuses), or feature change >=0.5. Keep branch 0.
All cached modules use one decision per UNet call. Solver history is unchanged.

## Fresh Paired Results

Three cases, four modes, one full warmup per mode, same initial noise per case,
512x512, FP16, guidance 7.5. Timings are CUDA synchronized and include full
pipeline execution and proxy overhead, but exclude setup and saving. SSIM and
PSNR compare with this run's uncached DPM20 images, not PNDM50.

| Mode | Mean seconds | Speedup vs DPM20 | Mean full calls | Mean SSIM | Mean PSNR dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPM20 | 61.18 | 1.00x | 20 | 1.0000 | infinite |
| Fixed interval 3 | 26.85 | 2.28x | 7 | 0.3490 | 12.16 |
| Lambda-only | 32.92 | 1.86x | 9 | 0.4419 | 13.53 |
| Adaptive V1 | 34.53 | 1.77x | 9.67 | 0.4550 | 14.07 |

Adaptive V1 is about 28.6% slower than fixed cache in this run. It uses 10, 10,
and 9 full calls, versus 7 for fixed cache. Its peak PyTorch allocated memory is
3642.62 MiB, versus 3642.00 MiB for fixed cache and 3533.93 MiB for uncached.
Neither the extra refresh count nor the extra proxy overhead is excluded from
the timing. Timing ranges and per-image metrics are saved in JSON/CSV.

## Per-Case SSIM

| Case | Fixed | Lambda-only | Adaptive |
| --- | ---: | ---: | ---: |
| Astronaut 42 | 0.2883 | 0.2574 | 0.2401 |
| Astronaut 43 | 0.2476 | 0.3391 | 0.3983 |
| Landscape 42 | 0.5110 | 0.7292 | 0.7265 |

Average reference similarity increases, but astronaut 42 regresses even with
more computation. On the equal-full-call landscape case, the feature-aware
method is slightly below lambda-only SSIM. The small average advantage over
lambda-only cannot establish that the feature signal is a better allocator.

Visual inspection shows that all outputs retain broad prompt content. The
landscape water color is closer to the reference than fixed cache. Astronaut
placement, suit details and background still drift; the first case remains
visibly different. These are observations about reference fidelity, not a
general aesthetic or semantic quality ranking.

## Observed Refresh Decisions

- Fixed: `[0,3,6,9,12,15,18]` for all cases.
- Lambda-only: `[0,1,3,6,9,12,15,18,19]` for all cases.
- Adaptive astronauts: `[0,1,3,5,7,10,13,16,18,19]`.
- Adaptive landscape: `[0,1,3,5,8,11,14,17,19]`.

Risk triggers fire four times for each astronaut and twice for the landscape.
The feature-spike guard never fires. Thus feature-weighted risk changes
decisions and the observed schedule differs by content in this pilot.

## Validation and Remaining Work

Seven CPU behavior tests pass, covering accumulated risk, feature influence,
guards, pooled-probe ordering, actual module skipping, restoration, and exact
uncached equivalence when all calls refresh on a toy UNet. Real SD1.5 runs
completed for both threshold settings. All 12 cases in this run have matching
paired noise hashes, 20 calls and false safety flags. Output artifacts include
source hashes, actual scheduler class, per-step decisions and a contact sheet.

No large dataset, repeated timing trials, CLIP, LPIPS or FID were evaluated.
No oracle measurements of true cache/solver error were collected. No equal
latency or matched-refresh-count controls were run against fixed caching.
Next validation should test those controls and unseen prompts before selecting
thresholds or claiming an adaptive-method advantage. The current code is a
working first version with measured limitations; no additional tuning was done.
