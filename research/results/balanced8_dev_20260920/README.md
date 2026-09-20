# Balanced eight-call development comparison

This experiment tests whether the five-step gap in `drop-middle8` can be repaired
without adding a ninth full UNet call. It uses the 10 analysis prompts and two
seeds. Previously consumed held-out prompts are not used.

```text
drop-middle8: [0, 2, 5, 7, 12, 14, 17, 19]  # existing reference, max gap 5
balanced8-A:  [0, 2, 5, 7, 10, 13, 16, 19]  # max gap 3
balanced8-B:  [0, 2, 5, 8, 11, 14, 17, 19]  # max gap 3
```

Run from the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/balanced8_dev_20260920/run_balanced.py --output research/results/balanced8_dev_20260920/run
```

Use `--resume` after interruption. Use `--resume --repair-nsfw` only if the safety
checker replaced an output with black.
