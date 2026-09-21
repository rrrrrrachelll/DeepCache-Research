# Block Feature Error Profiling + Activation Replacement Oracle

Cases: 10. Schedule: balanced8-A. Replacement probes: age-2 reuse steps.

## Overall block results

| Rank | Block | Role | Age-2 feature rel-L1 | Recovery fraction | Replacement guided error | Suffix ms |
|---:|---|---|---:|---:|---:|---:|
| 1 | down_block_0 | control | 0.232955 | 1.0000 | 0.000000 | 2158.9 |
| 2 | up_boundary | control | 0.265786 | 1.0000 | 0.000000 | 423.2 |
| 3 | up_block_1 | candidate | 0.343373 | 0.6858 | 0.035590 | 1416.8 |
| 4 | down_block_1 | candidate | 0.329875 | 0.6395 | 0.040570 | 1997.1 |
| 5 | up_block_2 | candidate | 0.369061 | 0.4574 | 0.059560 | 1672.2 |
| 6 | down_block_2 | candidate | 0.385813 | 0.4361 | 0.061652 | 1913.7 |
| 7 | up_block_3 | candidate | 0.345037 | 0.1879 | 0.086867 | 1816.3 |
| 8 | down_block_3 | candidate | 0.382355 | 0.1876 | 0.086875 | 1887.8 |
| 9 | mid_block | candidate | 0.374448 | 0.1727 | 0.088296 | 1864.5 |

## Actionable recovery by denoising step

| Step | up_block_1 | down_block_1 | up_block_2 | down_block_2 | up_block_3 | down_block_3 | mid_block |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.8539 | 0.7993 | 0.6361 | 0.5924 | 0.2765 | 0.2726 | 0.2337 |
| 9 | 0.7775 | 0.7392 | 0.5364 | 0.5157 | 0.1904 | 0.1945 | 0.1833 |
| 12 | 0.6842 | 0.6418 | 0.4455 | 0.4285 | 0.1462 | 0.1462 | 0.1391 |
| 15 | 0.6074 | 0.5617 | 0.3707 | 0.3550 | 0.1565 | 0.1557 | 0.1441 |
| 18 | 0.5057 | 0.4557 | 0.2983 | 0.2892 | 0.1699 | 0.1692 | 0.1631 |

## Interpretation

- Mean baseline guided-noise error over all reuse steps: 0.077353.
- `up_block_1` is the strongest actionable target overall and at every probed denoising stage; `down_block_1` is consistently second.
- The deepest candidates (`up_block_3`, `down_block_3`, and `mid_block`) have large feature errors but recover less than 19% of output error on average. Feature-error magnitude alone is therefore not a sufficient block-selection proxy.
- Recovery from the leading candidates declines toward later steps, so a correction/recompute rule should include timestep or log-SNR stage rather than one global threshold.
- `down_block_0` and `up_boundary` are endpoint upper-bound controls. Their 100% recovery is expected and does not establish a deployable speedup.

Recovery fraction is `(baseline error - replacement error) / baseline error`; larger is better. Suffix time measures the oracle probe only and excludes the cost of obtaining the teacher activation.
The oracle is teacher-forced and local to each denoising step. It ranks causal block repair value but does not by itself establish final-image quality or deployment speed.
