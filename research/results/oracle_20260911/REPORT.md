# Oracle Cache Error Analysis

10 analysis prompts, 20 images, 260 reuse observations.

Refresh steps are excluded from all correlations. Confidence intervals resample whole prompts.

| Target | Proxy | Spearman | 95% prompt CI | Within-step rank correlation |
|---|---|---:|---|---:|
| boundary_error | conv_previous | 0.232 | [0.052, 0.413] | -0.007 |
| boundary_error | conv_cache | 0.635 | [0.521, 0.757] | -0.089 |
| boundary_error | delta_lambda | -0.046 | [-0.210, 0.109] | N/A (constant within step) |
| boundary_error | lambda_distance | 0.562 | [0.425, 0.697] | N/A (constant within step) |
| boundary_error | age | 0.706 | [0.646, 0.772] | N/A (constant within step) |
| mid_error | conv_previous | 0.161 | [0.006, 0.295] | -0.033 |
| mid_error | conv_cache | 0.391 | [0.280, 0.486] | -0.101 |
| mid_error | delta_lambda | 0.239 | [0.154, 0.321] | N/A (constant within step) |
| mid_error | lambda_distance | 0.512 | [0.470, 0.563] | N/A (constant within step) |
| mid_error | age | 0.512 | [0.492, 0.542] | N/A (constant within step) |
| guided_noise_error | conv_previous | 0.018 | [-0.177, 0.237] | -0.058 |
| guided_noise_error | conv_cache | 0.291 | [0.187, 0.442] | -0.116 |
| guided_noise_error | delta_lambda | 0.011 | [-0.232, 0.250] | N/A (constant within step) |
| guided_noise_error | lambda_distance | 0.337 | [0.179, 0.529] | N/A (constant within step) |
| guided_noise_error | age | 0.333 | [0.287, 0.413] | N/A (constant within step) |

## Interpretation Limits

- Fixed interval 3, uncached teacher trajectory only
- Repeated timesteps are not independent samples; CI resamples prompts
- Within-step rank correlation removes shared timestep trends
- Exploratory analysis, not validation of an adaptive policy
- Held-out prompts are not generated or analyzed
- The saved images are uncached reference images, not adaptive-policy results.
- Diagnostic runtime includes oracle passes, hooks and reductions; it is not an inference speed benchmark.
- Local guided-noise error is not final perceptual image quality or a solver truncation-error estimate.
