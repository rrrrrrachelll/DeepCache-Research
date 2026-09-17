# One-time held-out final validation

The policy, methods, and acceptance criteria are frozen in `PROTOCOL.md`. This
experiment evaluates five previously unused prompts with two seeds and must not
be used as a new development sweep.

Run from the repository root:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/held_out_final_20260917/run_final.py --output research/results/held_out_final_20260917/run
```

Use `--resume` after interruption. Use `--resume --repair-nsfw` only when the
safety checker replaced a measured image with black.

Pre-run smoke testing must use a development prompt:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/held_out_final_20260917/run_final.py --output /tmp/held_out_final_smoke --smoke-analysis --limit 1
```
