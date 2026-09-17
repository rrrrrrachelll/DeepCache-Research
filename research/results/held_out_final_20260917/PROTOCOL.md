# Frozen held-out validation protocol

This is the one-time final evaluation on the five previously unused prompts and
two existing seeds. No policy choice may be changed after reading these results.

## Compared methods

- `dpm20`: uncached 20-call reference
- `fixed3`: `[0, 3, 6, 9, 12, 15, 18]`
- `uniform9`: `[0, 2, 5, 7, 10, 12, 14, 17, 19]`
- `stage_aware9`: `[0, 4, 7, 10, 13, 14, 16, 18, 19]`
- `drop_middle8`: `[0, 2, 5, 7, 12, 14, 17, 19]`

All methods use SD1.5, DPM-Solver++ with 20 steps, guidance scale 7.5,
`cache_branch_id=0`, and matched initial latents.

## Frozen primary decision

The primary contrast is paired final-image SSIM for `drop_middle8 - uniform9`,
grouped by prompt so the two seeds remain together.

`drop_middle8` is accepted as the lower-cost final candidate only if all of the
following hold:

1. Its mean SSIM loss relative to `uniform9` is at most 0.01.
2. At most one of the five prompt groups loses more than 0.03 SSIM.
3. Every trace has exactly eight full calls, and its mean runtime is lower than
   `uniform9`.

PSNR, RGB error, fixed3, and stage-aware9 are secondary diagnostics. The held-out
prompts will not be reused to tune another replacement policy.

If a safety-checker false positive replaces an image with black, only that exact
prompt, seed, and method is regenerated with image replacement disabled. Such a
repair does not change the frozen schedule or latent.
