# Stage-aware V2 closed-loop development experiment

This experiment uses the existing 10 analysis prompts and two seeds. The five held-out prompts remain unused.

The primary comparison holds compute constant at nine complete UNet calls:

- `uniform9`: refresh at `[0, 2, 5, 7, 10, 12, 14, 17, 19]`
- `stage_aware9`: refresh at `[0, 4, 7, 10, 13, 14, 16, 18, 19]`

`stage_aware9` allocates 2/3/4 refreshes to steps 0–6, 7–13, and 14–19. Uncached DPM20 and seven-refresh fixed interval 3 are included as references.

Run from the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/stage_aware_v2_dev_20260915/run_stage_aware.py --output research/results/stage_aware_v2_dev_20260915/run
```

If execution stops after the output directory has been created, resume with `--resume`. After completion, run `analyze_results.py` in the isolated Matplotlib environment.
