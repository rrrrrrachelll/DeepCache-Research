# Current-Step Spatial Feature Conditioned Residual Predictability

LOPO cases: 20; probe rows: 600. Best deployable model: `shallow_delta`.

| Model | Residual rel-L1 | Corrected feature rel-L1 | Guided error | Recovery | Oracle gain | Positive prompts | Probe ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached_lowrank | 0.9665 | 0.3628 | 0.122348 | 0.0134 | 0.020 | 8/10 | 1389.8 |
| conv_delta | 0.9667 | 0.3624 | 0.122127 | 0.0154 | 0.023 | 8/10 | 1388.5 |
| shallow_delta | 0.9651 | 0.3620 | 0.121202 | 0.0245 | 0.036 | 9/10 | 1386.9 |
| combined | 0.9657 | 0.3622 | 0.121487 | 0.0212 | 0.032 | 9/10 | 1388.1 |
| combined_spatial_shuffle | 0.9792 | 0.3682 | 0.121953 | 0.0172 | 0.025 | 8/10 | 1383.7 |
| oracle | 0.0000 | 0.0000 | 0.041884 | 0.6735 | 1.000 | 10/10 | 1397.2 |

## Recovery by step

| Step | Cached only | Conv delta | Shallow delta | Combined | Spatial shuffle | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | -0.0716 | -0.0710 | -0.0449 | -0.0567 | -0.0582 | 0.8609 |
| 9 | -0.0268 | -0.0191 | -0.0101 | -0.0108 | -0.0210 | 0.7517 |
| 12 | 0.0170 | 0.0183 | 0.0236 | 0.0217 | 0.0182 | 0.6874 |
| 15 | 0.0539 | 0.0527 | 0.0572 | 0.0547 | 0.0537 | 0.5893 |
| 18 | 0.0944 | 0.0961 | 0.0966 | 0.0972 | 0.0931 | 0.4780 |

## Predeclared decision

| Gate | Pass |
|---|---:|
| mean_recovery_at_least_20pct | no |
| at_least_7_positive_prompts | yes |
| step4_nonnegative | no |
| step9_nonnegative | no |
| combined_beats_spatial_shuffle | yes |

**FAIL.** No current-conditioned rank-16 model reaches the predeclared 20% recovery threshold, and the best model still worsens steps 4 and 9.

The aligned combined model exceeds its spatial-shuffle control by only 0.41 percentage points. This is not strong evidence that the predictor learned a causal spatial correction.

The teacher oracle remains high (67.35%), so the repair location still has theoretical value. The failure is the generalization of the tested inference-available predictors, not the absence of a correct activation at this location.

Following the predeclared protocol, do not proceed to a closed-loop SSIM experiment or increase predictor depth. Stop this `up_block_1` residual-predictor branch and move to local selective recomputation or a cache representation that retains current spatial information.

All measurements are teacher-forced single-step probes. They do not directly measure final-image SSIM.
