# Initial V1 Diagnostic: Risk Threshold 0.65

This run measured three paired cases per method, with one warmup per method.
All similarities use this run's fresh DPM20 images as references.

| Method | Mean seconds | Full calls | SSIM | PSNR dB |
| --- | ---: | ---: | ---: | ---: |
| DPM20 | 59.66 | 20 | 1.0000 | infinite |
| Fixed interval 3 | 26.32 | 7 | 0.3490 | 12.16 |
| Lambda-only, tau=0.65 | 29.25 | 8 | 0.4117 | 13.32 |
| Adaptive V1, tau=0.65 | 29.35 | 8 | 0.4117 | 13.32 |

## Diagnostic Finding

All lambda-only and adaptive cases refreshed at indices
`[0, 1, 4, 7, 10, 13, 16, 19]`. All refreshes were triggered by startup,
maximum age or final-step guards. Risk and feature-spike triggers did not fire.
The lambda-only and feature-aware schedules and outputs are identical.
This setting therefore does not demonstrate content-adaptive behavior.

The improvement over fixed interval 3 is confounded by an extra full call and
different guarded refresh placement. It is not evidence for the feature proxy.

## Follow-up Calibration

The astronaut seed-42 trace shows risk 0.62965 at index 3 (cache age 2), just
below 0.65, followed by a maximum-age refresh at index 4. Based on this observed
risk scale, a follow-up run uses the explicit flag `--risk-threshold 0.50`.
This is calibration on observed cases, not a held-out evaluation or a search
result establishing the best threshold. The initial run is retained unchanged.

Source, environment and policy metadata are in `manifest.json`; full per-call
decisions are in `*_steps.json`. No CLIP/LPIPS/FID evaluation was performed.
