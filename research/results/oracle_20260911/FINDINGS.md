# Findings: Conv-In Versus True Cache Error

## Completed Experiment

SD1.5, DPM-Solver++ order 2 midpoint, 20 steps, fixed DeepCache interval 3,
branch 0, CFG 7.5, fp16, 512x512. Ten new analysis prompts and two seeds produce
20 images and 400 step records. All 260 reuse records enter the correlations;
140 refresh records are excluded. Five reserved prompts remain untouched.

The complete UNet and shadow DeepCache see exactly the same latent, timestep
and text conditioning. Only the complete prediction advances the scheduler.
Oracle computations cannot update cached tensors on reuse steps. The primary
deep error measures the cached boundary tensor, not merely the mid-block.

## Main Results

| Signal | Spearman with boundary error | Spearman with guided prediction error | Within-step boundary rank correlation |
|---|---:|---:|---:|
| Consecutive conv_in change (V1 proxy) | 0.232 | 0.018 | -0.007 |
| Conv_in change since last refresh | 0.635 | 0.291 | -0.089 |
| Lambda distance since refresh | 0.562 | 0.337 | Undefined: constant within each step |
| Cache age | 0.706 | 0.333 | Undefined: constant within each step |

The V1 proxy has only weak pooled association with the primary deep error and
almost no pooled association with guided prediction error. Its prompt-cluster
95% interval is [0.052, 0.413] for deep error and [-0.177, 0.237] for guided
prediction error. Within-step ranking provides no positive evidence that it
reliably distinguishes image-dependent deep errors in this experiment.

Using the last-refresh anchor improves pooled deep-error correlation to 0.635,
but this is not sufficient evidence of better content awareness. Holding cache
age fixed, its correlations fall to 0.161 at age 1 and 0.086 at age 2. Its
within-step association is slightly negative. Much of the pooled association
is therefore consistent with shared age/stage effects rather than useful
content discrimination. This is an interpretation, not a causal identification.

Cache age has the largest pooled deep-error correlation among the tested
signals. Overlapping uncertainty intervals mean these point estimates alone
are not a significance test of one predictor outperforming another.

## Implications for V2

Do not justify the existing `delta_lambda * (1 + beta * conv_change)` rule as
an accurate local-error estimator based on these results. In particular, simply
increasing beta or substituting the last-refresh conv_in difference is not yet
supported as a reliable content-adaptive improvement.

Keep age and lambda-only schedules as necessary controls. Next diagnostics
should test additional ages/intervals and cached rollout trajectories, then
evaluate whether an alternative low-cost feature signal adds predictive value
beyond age and stage. Any proxy selection/calibration belongs to the analysis
set; the reserved prompts must remain unseen until the chosen policy is fixed.
Evaluate that policy against equal-compute controls before claiming a speed or
quality improvement. No policy or threshold was changed in this experiment.

## Verification and Cost

- Fourteen combined new/existing unit tests passed.
- A paired real-model image exactly matched the pristine unwrapped DPM20 image.
- Toy-network shadow predictions matched the original DeepCache helper.
- All refresh-step feature and prediction errors were exactly zero.
- All 20 samples completed; none was flagged by the safety checker.
- Recorded diagnostic time: 1302.16 seconds total, 65.11 seconds per sample.
  This excludes the separate pristine check, loading and analysis. It includes
  expensive oracle instrumentation and is NOT an acceleration benchmark.
- Peak allocated GPU memory: approximately 3634 MiB per sample.
- The runner, helper and prompt hashes match their start-of-run manifest.
- Original baseline/adaptive files and earlier results were not modified.

## Limits and Artifacts

Only ten prompts, two shared seeds, one fixed interval and the uncached teacher
trajectory are covered. These are exploratory findings, not a universal claim
that shallow features cannot work. A different feature, metric, operating point
or trajectory could give a different result. Local prediction error is neither
perceptual image quality nor a DPM solver truncation-error estimate.

`REPORT.md` contains the full table; `analysis.json` includes per-age, per-stage,
per-prompt statistics and bootstrap intervals. `steps.csv` contains all raw
scalar measurements. `proxy_scatter.png` and `error_by_step.png` visualize the
results. Saved sample PNGs are uncached reference images, not adaptive outputs.
The manifest's serialized scheduler configuration can retain the model's
inherited class-name metadata; the runtime scheduler is explicitly constructed
and checked as DPMSolverMultistepScheduler by the runner/helper.
