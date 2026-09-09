# SD1.5 Sampling and Fixed Cache Pilot

Completed 12 measured generations (three paired cases across four methods),
plus one warmup per method, on Tesla M60 8 GiB. No adaptive method was added.
Original `main.py`, `DeepCache/`, and top-level baseline images were untouched.

## Results

| Method | Mean seconds | Observed range | Speedup | Peak allocated MiB | PSNR vs PNDM50 | SSIM vs PNDM50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| PNDM50 | 156.93 | 151.54-160.25 | 1.00x | 3526.84 | infinite | 1.0000 |
| DPM20 | 61.40 | 58.64-63.33 | 2.56x | 3526.81 | 15.83 | 0.5812 |
| PNDM50 + cache | 67.33 | 64.94-68.54 | 2.33x | 3635.28 | 17.21 | 0.6272 |
| DPM20 + cache | 27.26 | 26.62-27.62 | 5.76x | 3635.50 | 11.95 | 0.3565 |

The combined method is approximately 2.25x faster than DPM20 alone, but the
resulting images differ visibly. PNDM50 has 51 actual UNet calls because of a
repeated timestep. With the original helper, 17 are full calls and 34 reuse
features. DPM20 has 20 calls; with caching, 7 are full and 13 reuse features.
All paired initial latent hashes match. All safety flags were false.

## Isolating Cache Changes Under DPM20

| Case | PSNR of DPM20+cache vs DPM20 | SSIM |
| --- | ---: | ---: |
| Astronaut, seed 42 | 10.61 | 0.2883 |
| Astronaut, seed 43 | 9.84 | 0.2476 |
| Landscape, seed 42 | 16.02 | 0.5110 |

Visual inspection of `comparison.png` shows that all methods retain the broad
prompt content. The combined method changes astronaut scale/placement and
lighting, particularly at seed 43, and changes the landscape water color and
tree details. These observations establish output drift, not an absolute
ranking of aesthetic quality. Fixed-cache PNDM outputs are closer to the
reference by average SSIM/PSNR in this small pilot.

## Interpretation and Limits

The combined speed benefit is demonstrated on these cases. The hypothesis that
an adaptive strategy can retain more of DPM20's output while keeping much of
this benefit remains untested. No conclusion about a publishable new method
follows from the combination alone.

Two prompts and three prompt/seed pairs are insufficient for broad quality
claims. There are no repeated timings of the same pair, confidence intervals,
CLIP scores, LPIPS scores or FID measurements. SSIM/PSNR use the uncached output
as a reference, not real-image ground truth. Matching a reference is distinct
from prompt alignment or visual appeal. Timings should only be compared within
this run, not directly against the earlier 149.36/63.25-second demo.

See `manifest.json` for exact scheduler configurations, model snapshot,
versions, runner hash and timing boundaries. `metrics.csv` and `metrics.json`
contain per-image measurements; `*_steps.json` contain actual call traces.
The executable protocol is documented in `research/README.md`.
