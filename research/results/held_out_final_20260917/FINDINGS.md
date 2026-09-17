# Final held-out findings

## Decision

The frozen rule rejects `drop_middle8` as the final replacement for `uniform9`.
Its paired held-out SSIM difference is -0.01203 with a prompt-grouped bootstrap
95% interval of [-0.02126, -0.00339]. The mean loss is slightly larger than the
pre-registered 0.01 limit, and the interval is entirely below zero.

The other frozen checks pass. `drop_middle8` uses exactly eight full calls, runs
in 29.11 seconds on average instead of 31.80 seconds for `uniform9`, and no prompt
group loses more than 0.03 SSIM. It is a useful optional speed setting, but it
does not meet the final default-quality criterion.

## Final policy

Use `uniform9 = [0, 2, 5, 7, 10, 12, 14, 17, 19]` as the selected default policy.
It reaches mean held-out SSIM 0.5777 at 1.88x speedup relative to DPM20. The
stage-aware schedule is rejected: with the same nine full calls it is slightly
slower and its held-out SSIM falls to 0.4719. `fixed3` remains the fastest tested
setting at 2.25x, with substantially lower SSIM of 0.5115.

`drop_middle8` may be reported separately as a 2.05x speed option with mean SSIM
0.5656, provided it is not described as equivalent to `uniform9` under the
frozen validation rule.

## Statistical boundary

The five held-out prompts have now been consumed. They can support this final
choice among the frozen candidates, but cannot be used to tune a new schedule
and then validate that new schedule. Any further proxy or policy development
requires either development-only reporting or a newly reserved validation set.
