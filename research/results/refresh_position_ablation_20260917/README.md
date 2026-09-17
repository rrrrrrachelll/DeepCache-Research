# Refresh-position ablation

This closed-loop development experiment measures the marginal final-image value
of one refresh at an early, middle, or late DPM20 step. It reuses the validated
uncached DPM20 and uniform9 outputs from the Stage-aware V2 development run.

Starting from uniform9:

```text
[0, 2, 5, 7, 10, 12, 14, 17, 19]
```

the three equal-budget eight-call modes remove exactly one refresh:

- `drop_early8`: delete step 2, leaving an anchor span from 0 to 5
- `drop_middle8`: delete step 10, leaving an anchor span from 7 to 12
- `drop_late8`: delete step 17, leaving an anchor span from 14 to 19

All spans have the same length. The experiment uses 10 development prompts and
two seeds; the five held-out prompts remain unused.

Run from the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/refresh_position_ablation_20260917/run_ablation.py --output research/results/refresh_position_ablation_20260917/run
```

Use `--resume` if a partial run directory already exists. If the safety checker
replaces a measured image with black, use `--resume --repair-nsfw` to regenerate
only flagged rows with image replacement disabled. After completion, run
`analyze_results.py` with the isolated Matplotlib dependencies. Final results and
interpretation are in `REPORT.md` and `FINDINGS.md`.
