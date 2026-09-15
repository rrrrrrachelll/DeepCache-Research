# Findings: Stage-aware refresh at nine full UNet calls

## Result

The tested stage-aware schedule is rejected on the development set.

Both `uniform9` and `stage_aware9` use exactly nine complete UNet calls. The
stage-aware schedule moves refresh budget from early sampling to late sampling:
it refreshes 2/3/4 times in steps 0–6, 7–13, and 14–19. Across ten development
prompts and two seeds, it is less similar to uncached DPM20 than uniform9:

| Metric | Uniform9 | Stage-aware9 | Stage-aware minus uniform |
| --- | ---: | ---: | ---: |
| SSIM | 0.5481 | 0.4665 | -0.0816 |
| PSNR | 15.391 dB | 13.832 dB | -1.559 dB |
| Mean absolute RGB error | 0.1209 | 0.1471 | +0.0261 |
| Runtime | 31.73 s | 32.02 s | +0.30 s |

All ten prompt groups favor uniform9 in SSIM, PSNR, and RGB error. The
prompt-bootstrap 95% interval for the SSIM difference is
`[-0.1155, -0.0538]`. Stage-aware9 is also worse than the seven-refresh fixed3
baseline on average despite using two additional complete calls.

## Interpretation

The Oracle interval sweep established that local boundary and guided-noise
errors grow with cache age and become especially large late in sampling. This
closed-loop experiment shows that minimizing those late local errors alone is
not a sufficient allocation rule for final-image fidelity. A plausible
explanation is that errors introduced early alter the latent trajectory and
propagate through all later steps, while late errors have fewer opportunities
to affect global structure. The current experiment does not directly identify
that causal mechanism.

This result rejects the specific `[0, 4, 7, 10, 13, 14, 16, 18, 19]` schedule.
It does not establish that every stage-aware schedule is worse. In particular,
this schedule makes two consecutive refresh pairs and removes the early
refreshes at steps 2 and 5 used by uniform9, so its allocation may be too
aggressive.

## Next experiment

Keep the five held-out prompts untouched. On the development prompts, perform a
small, predeclared equal-budget refresh-position ablation around uniform9.
Move one refresh at a time from early or middle sampling to a late step, rather
than changing several positions simultaneously. This will estimate the
closed-loop value of refreshes at individual stages without a broad schedule
search. Select a new policy only if it improves uniform9 consistently across
prompt groups, then freeze it before held-out validation.
