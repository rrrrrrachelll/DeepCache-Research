# up_block_1 Residual Predictability

LOPO cases: 20; probe rows: 300.

| Model | Residual rel-L1 | Corrected feature rel-L1 | Guided error | Recovery | Oracle gain | Probe ms |
|---|---:|---:|---:|---:|---:|---:|
| channel_mean | 0.9695 | 0.3634 | 0.121086 | 0.0261 | 0.039 | 1379.1 |
| channel_affine | 0.9678 | 0.3631 | 0.119055 | 0.0421 | 0.062 | 1369.4 |
| oracle | 0.0000 | 0.0000 | 0.041884 | 0.6735 | 1.000 | 1396.5 |

## Recovery by step

| Step | Channel mean | Channel affine | Oracle |
|---:|---:|---:|---:|
| 4 | -0.0232 | -0.0022 | 0.8609 |
| 9 | -0.0172 | -0.0013 | 0.7517 |
| 12 | 0.0231 | 0.0350 | 0.6874 |
| 15 | 0.0538 | 0.0647 | 0.5893 |
| 18 | 0.0942 | 0.1140 | 0.4780 |

## Prompt-grouped result

Best deployable baseline: `channel_affine`; positive prompts: 10/10.

## Decision

FAIL: cached activation plus step-conditioned channel statistics are not sufficient; do not deploy this corrector.
