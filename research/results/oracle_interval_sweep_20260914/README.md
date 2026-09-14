# Oracle interval sweep

Reuses the completed [interval-3 oracle run](../oracle_20260911/) and generates only intervals 2, 4, and 5. All runs use the same 10 analysis prompts, two seeds, SD1.5, 20-step DPM-Solver++, branch 0, and paired full/shadow UNet on the same latent. Held-out prompts remain unused.

From the repository root, run:

```bash
/root/miniconda3/envs/deepcache/bin/python -B research/results/oracle_interval_sweep_20260914/run_sweep.py
```

Each interval writes its own immutable output directory. The runner skips a directory only after `complete.json` exists. The original Oracle runner now accepts `--interval`; its default remains 3.

After all three `complete.json` markers exist:

```bash
PYTHONPATH=/tmp/deepcache_sweep_plot_deps /root/miniconda3/envs/deepcache/bin/python -B research/results/oracle_interval_sweep_20260914/analyze_sweep.py
```

Install the pinned packages from `../oracle_20260911/oracle_analysis_requirements.txt` into an isolated analysis environment if needed. The analyzer checks protocol fields, latent hashes, and exact reference-image hashes across intervals before combining data. It excludes refresh rows, makes a conditional mean error heatmap by cache age and DPM step, and compares intercept-only, age, lambda-distance, and combined linear models using leave-one-prompt-out folds.
