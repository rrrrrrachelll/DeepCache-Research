# Adaptive DeepCache V1

This experimental controller changes refresh timing while retaining the original
SD1.5 branch-0 spatial skip boundaries. It supports second-order deterministic
`DPMSolverMultistepScheduler` with `algorithm_type="dpmsolver++"` and unique
integer timesteps. The validated setup is the existing SD1.5 20-step experiment.
It does not implement adaptive depth, feature prediction or solver-history resets.

## Decision

At each UNet call, before cached down blocks run, observe the fresh `conv_in`
output. Use 4x4 average pooling, then float32 reductions. Let E be mean absolute
change relative to the preceding call's pooled feature, divided by that previous
feature's mean absolute magnitude (denominator clamped to 1e-6).

The noise coordinate is lambda = log(alpha) - log(sigma) = 0.5 * log(SNR), using
the installed scheduler's VP alpha/sigma arrays. It is not log(alpha_cumprod).
Accumulate R += abs(lambda_current - lambda_previous) * (1 + beta * E).
On refresh, reset R to zero. Otherwise all cached modules reuse the same stored
outputs and all receive the same per-call refresh decision.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| risk_threshold | 0.65 | Refresh when accumulated R reaches this value |
| feature_weight | 2.0 | beta, weight of feature-change proxy |
| feature_spike | 0.5 | Immediate refresh if E reaches this value |
| max_age | 3 | Refresh at age 3, allowing at most two consecutive reuse calls |
| initial_full_steps | 2 | Full calls while the multistep solver starts |
| final_full_steps | 1 | Full final UNet call |

Rules are prioritized: initial, final, nonfinite proxy, feature spike, maximum
age, risk threshold, reuse. First-call E is zero. Feature_weight=0 disables both
the feature contribution and feature-spike trigger for the lambda-only ablation.
The ablation still measures/logs the feature proxy so instrumentation is shared.

These are untuned heuristic defaults, not fitted local-error estimates or
validated optimal thresholds. `conv_in` is before time/text-conditioned UNet
blocks, so this proxy does not directly measure their conditional feature error.
Pooling can hide fine spatial changes. Extra reductions include a GPU-to-CPU
scalar synchronization; this overhead is included in measured pipeline time.

The startup/final guards and max_age=3 require at least eight full calls in 20
steps, versus seven for original interval=3. Do not attribute quality gains to
better allocation without equal-cost/latency comparisons. Refreshing the UNet
does not remove approximate predictions already in the multistep solver history.

## Run

From the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B -m unittest research.test_adaptive_cache -v
/root/miniconda3/envs/deepcache/bin/python -B research/run_adaptive.py --output research/results/adaptive_new_run
```

Optional flags: `--risk-threshold`, `--feature-weight`, `--feature-spike`,
`--max-age`. An existing output directory is rejected to preserve prior results.

The runner measures fresh DPM20, fixed interval-3 cache, lambda-only scheduling
and adaptive V1 on the same three cases. Each mode has one full warmup. Cases
rotate mode order. Dimensions, precision, guidance and CUDA initial-noise
generation match `compare_sampling.py`. All pairs must have equal noise hashes.
The original safety checker stays enabled.

Pipeline timing synchronizes CUDA and includes text encoding, UNet, VAE,
safety checking, output conversion and adaptive instrumentation. Loading,
helper setup, disk writes and similarity evaluation are excluded. Memory is
peak PyTorch allocated memory, not total device usage.

Artifacts: `manifest.json` (explicit runtime scheduler class plus config and
source hashes), `run.log`, `metrics.json`, `metrics.csv`, `summary.json`, per-step
JSON traces and PNGs, and a four-column `comparison.png`. Nonfinite scheduler
metadata values are strings so JSON is standards-compliant.

SSIM/PSNR compare each method with the fresh DPM20 output, not PNDM50 or ground
truth. Identical-image infinite PSNR is encoded as null. No CLIP, LPIPS or FID
is computed. Three paired cases cannot establish generalization or novelty.

## Integration Contract

Create a helper per generation, after `scheduler.set_timesteps(steps)`, call
`enable()`, run the pipeline, and call `disable()` in a `finally` block. The
pipeline may reset the same timestep schedule before denoising. Changed or
repeated schedules and accidental helper reuse across generations are rejected.
Trace records remain available after disable; feature tensors and hooks do not.
The helper is intended for sequential inference, not training, compilation or
concurrent calls to the same pipeline.
