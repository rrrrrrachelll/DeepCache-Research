# Sampling and Fixed Cache Pilot

Run from the repository root using the existing environment:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/compare_sampling.py --output research/results/new_run
```

The output directory must not exist. The runner uses local SD1.5 weights and
the unchanged `DeepCacheSDHelper`. It does not implement adaptive caching.

| Mode | Scheduler | Requested steps | Cache |
| --- | --- | --- | --- |
| pndm50 | Original model scheduler (PNDM) | 50 | Off |
| dpm20 | DPM-Solver++, order 2, midpoint, multistep, linspace | 20 | Off |
| pndm50_cache | Original model scheduler | 50 | Interval 3, branch 0 |
| dpm20_cache | Same DPM-Solver++ | 20 | Interval 3, branch 0 |

The three paired cases comprise two prompts, with two seeds for the astronaut
prompt and one seed for the landscape prompt. Initial CUDA float16 noise is
identical across methods within each case, verified by SHA256. Image size is
512x512, guidance is 7.5, and the original safety checker remains enabled.
Each mode receives one full warmup. Measurement order rotates across cases.

Timing uses CUDA synchronization and includes the full pipeline and CPU output
conversion, but excludes loading, helper setup, saving and quality evaluation.
Peak allocated memory is PyTorch allocation, not total GPU memory usage. Actual
UNet calls and refresh decisions are recorded because PNDM can repeat timesteps.
Classifier-free guidance uses a batched UNet call; NFE here counts calls, not
individual conditional/unconditional batch elements.

Outputs include individual PNGs, a labeled contact sheet, per-call traces,
environment/scheduler metadata, per-image JSON/CSV metrics and aggregate JSON.
PSNR and SSIM compare saved RGB images with the paired uncached PNDM50 image;
an additional comparison isolates caching changes relative to DPM20. SSIM uses
an 11x11 Gaussian window, sigma 1.5, population moments and valid cropping,
averaged across RGB channels. Identical-image PSNR is stored as null (infinite).

This is a small exploratory pilot. Similarity to PNDM is not a measure of
absolute image quality or prompt alignment. No CLIP, LPIPS or FID is computed.
Reported timing ranges reflect different cases, not repeated measurements of
one case. Inspect the PNGs and safety flags before interpreting quality scores.
Later research evaluation should increase prompts/seeds and repeat timings.
