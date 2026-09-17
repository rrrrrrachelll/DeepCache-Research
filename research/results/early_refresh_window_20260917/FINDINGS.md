# Findings

## Main result

The existing step-2 placement is the strongest default for the second full-UNet
refresh. Its mean SSIM is 0.5481, compared with 0.5281 at step 1, 0.5283 at step
3, and 0.4861 at step 4. All schedules use the same nine full calls and keep all
later refreshes fixed.

Relative to step 2, the mean SSIM differences are -0.02005 for step 1 (95% CI
[-0.04971, 0.00815]), -0.01976 for step 3 (95% CI [-0.04717, 0.00711]), and
-0.06199 for step 4 (95% CI [-0.10571, -0.02596]). Step 1 and step 3 are not
distinguishable from step 2 or from each other at this sample size. Step 4 is
reliably worse: it loses on 9 of 10 prompt groups, and its paired interval against
steps 1, 2, and 3 excludes zero.

## Interpretation

The early refresh requirement is a window rather than a monotonic preference for
the earliest possible step. Refreshing again at step 1 is often too soon to add
useful trajectory correction, while waiting until step 4 allows the initial
cached approximation to propagate too far. Step 2 is the best observed balance
and has the highest aggregate SSIM, PSNR, and lowest RGB error.

This result also isolates one cause of the failed stage-aware schedule: that
schedule moved the second refresh from step 2 to step 4. The controlled sweep
shows that this move alone reduces final-image fidelity even when the total call
budget and every later refresh are unchanged.

## Policy decision and next experiment

Keep `[0, 2, 5]` as a fixed early scaffold. Do not let an age or local-error proxy
postpone the second refresh beyond step 3. The prior position ablation found that
the step-10 refresh contributes little, so `drop_middle8 = [0, 2, 5, 7, 12, 14,
17, 19]` is now the strongest lower-cost candidate. The next useful experiment
is a one-time held-out comparison of `uniform9` and `drop_middle8` against DPM20
to test whether the saved call generalizes before adding a more complex proxy.

Held-out prompts were not used in this sweep.
