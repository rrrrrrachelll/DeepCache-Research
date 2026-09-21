# up_block_1 Mechanism Ablation

Cases: 10. Probes: 150. All probes use the same teacher-forced DPM20 trajectory.

| Probe | Recovery fraction | Guided-noise error | Probe time (ms) |
|---|---:|---:|---:|
| activation_only | 0.0000 | 0.106415 | 419.1 |
| suffix_only | 0.0276 | 0.103017 | 1380.8 |
| activation_plus_suffix | 0.6858 | 0.035590 | 1394.8 |

## Recovery by denoising step

| Step | activation only | suffix only | activation + suffix |
|---:|---:|---:|---:|
| 4 | 0.0000 | -0.0054 | 0.8539 |
| 9 | 0.0000 | -0.0026 | 0.7775 |
| 12 | 0.0000 | 0.0231 | 0.6842 |
| 15 | 0.0000 | 0.0474 | 0.6074 |
| 18 | 0.0000 | 0.0758 | 0.5057 |

## Runtime implication

Five full-suffix probes add approximately **6.97 s** of GPU work per image. Added to the existing balanced8-A mean (29.84 s), this gives a lower-bound estimate of **36.81 s**, already slower than uniform9 (31.73 s) before paying for a residual predictor.

## Decision

Activation correction and suffix recomputation are complementary. Do not train the residual predictor yet; first scan shorter propagation cut points after up_block_1.

The next experiment should keep teacher activation replacement as the upper bound and vary how many downstream modules are recomputed, measuring recovery per millisecond.

Probe time is an oracle-pass diagnostic, not end-to-end deployment latency. Teacher activations are unavailable at inference.
