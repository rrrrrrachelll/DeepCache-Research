# Early refresh-window sweep

This closed-loop development experiment locates the best position for the second
full-UNet refresh while holding the compute budget and every later refresh fixed.

The four schedules are:

```text
early1: [0, 1, 5, 7, 10, 12, 14, 17, 19]
early2: [0, 2, 5, 7, 10, 12, 14, 17, 19]  # existing uniform9 reference
early3: [0, 3, 5, 7, 10, 12, 14, 17, 19]
early4: [0, 4, 5, 7, 10, 12, 14, 17, 19]
```

All schedules use nine full calls. The experiment measures `early1`, `early3`,
and `early4` on the 10 development prompts and two seeds, and reuses the paired
`uniform9` data for `early2`. The five held-out prompts remain unused.

Run from the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/early_refresh_window_20260917/run_sweep.py --output research/results/early_refresh_window_20260917/run
```

Use `--resume` for an interrupted run. If the safety checker replaces a measured
image with black, use `--resume --repair-nsfw` to regenerate only flagged rows
with image replacement disabled.
