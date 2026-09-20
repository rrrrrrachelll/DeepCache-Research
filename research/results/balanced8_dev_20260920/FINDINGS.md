# Findings

## Main result

`balanced8_a = [0, 2, 5, 7, 10, 13, 16, 19]` is the best observed strict
eight-call schedule. Its mean SSIM is 0.54385, compared with 0.54323 for
`drop_middle8` and 0.53976 for `balanced8_b`. Mean runtimes are similar because
all three schedules use exactly eight full UNet calls.

The improvement of balanced8-A over drop-middle8 is small: +0.00062 SSIM with a
prompt-grouped 95% bootstrap interval of [-0.00374, 0.00424]. It improves 7 of
10 prompt groups, but the interval includes zero. Removing the five-step gap is
therefore directionally useful but not sufficient evidence of a large fidelity
gain.

Balanced8-A is stronger than balanced8-B by +0.00410 SSIM, although the SSIM
interval [-0.00034, 0.00903] narrowly includes zero. Its PSNR advantage and RGB
error reduction have intervals above zero, so the A phase is the better balanced
layout for continued evaluation.

## Relation to uniform9

Balanced8-A trails uniform9 by 0.00425 SSIM on average, with a 95% interval of
[-0.00672, -0.00120]. The loss is small but consistent across 9 of 10 prompt
groups. This makes balanced8-A a stronger development candidate for the fast
mode, while uniform9 remains the default quality-speed policy.

The experiment does not show that max-gap minimization alone predicts final
quality: balanced8-B also has maximum gap three, yet performs worse. Refresh
phase and diffusion stage remain important.

## Decision

Carry `balanced8_a` forward as the preferred strict eight-call candidate and
retire `balanced8_b`. Do not claim final generalization from this development
comparison. Further confirmation requires a newly reserved validation set; the
previous held-out prompts have already been consumed and were not reused here.
