# Conv-In Proxy / Deep Cache Error Experiment

This isolated experiment leaves `main.py`, `DeepCache/`, adaptive-v1 and all
previous results unchanged. It diagnoses a proxy; it does not implement V2.

## Protocol

- SD1.5, fp16, 512x512, CFG 7.5, DPM-Solver++ order 2 midpoint, 20 steps,
  linspace timesteps, no Karras sigmas, fixed DeepCache interval 3 / branch 0.
- Ten new analysis prompts times two seeds. Five additional prompts are reserved
  and are NOT generated or used to choose a proxy/threshold.
- At each step the complete UNet computes the oracle at the current reference
  latent. On reuse steps, a second shadow forward uses DeepCache at exactly that
  same latent, timestep and conditioning. Only the full prediction is returned
  to the sampler. Oracle passes cannot read or overwrite shadow caches.
- On refresh steps the full pass itself populates the cache, so no duplicate
  full pass is necessary. Each image requires 20 full and 13 partial forwards.
- The first image is also generated without instrumentation. Exact equality of
  the resulting float images is asserted, and hooks are removed after each case.

## Measurements

`conv_previous` matches v1: relative mean absolute change between consecutive
4x4-average-pooled conv_in outputs, divided by previous feature magnitude.
`conv_cache` instead compares the same fresh pooled feature with its value at
the last refresh, divided by the last-refresh magnitude.

The primary deep tensor is `up_blocks[-1].attentions[-2]` output, corresponding
to helper key `('up', 'attentions', 0, 1)`. At branch 0 this is the cached tensor
feeding the final freshly computed resnet/attention pair. It is a deep-path
output, despite its location near the UNet output. Mid-block output is a
secondary probe. Errors use `mean(abs(cached-full))/max(mean(abs(full)),1e-6)`.
Both CFG branches and their aggregate are recorded. Noise error is measured
before and after guidance, on the same input rather than different trajectories.

Lambda is half log-SNR: `log(alpha)-log(sigma)`. Record its step increment,
distance since refresh, and cache age alongside the feature proxies.

Refresh-step zero errors are excluded from correlations. Reports include
Pearson/Spearman correlations, 1,000 prompt-cluster bootstrap intervals,
per-age/per-stage/per-prompt correlations, and within-step centered-rank
correlations. The latter distinguish image-dependent information from shared
sampling-stage trends. No naive independent-step significance tests are used.

## Run

```bash
/root/miniconda3/envs/deepcache/bin/python -B -m unittest research.test_oracle_cache
/root/miniconda3/envs/deepcache/bin/python -B research/run_oracle.py --output research/results/oracle_NEW
PYTHONPATH=/tmp/deepcache-oracle-analysis /root/miniconda3/envs/deepcache/bin/python -B research/analyze_oracle.py research/results/oracle_NEW
```

The analysis requires SciPy 1.13.1 and Matplotlib 3.8.4, installed separately
from the baseline environment for this run. `--limit 1` is a smoke test, not the
full experiment. Existing output directories are never overwritten by the runner.
Each completed case saves its trace and image immediately. `complete.json`
marks successful completion; the analyzer requires it.

## Interpretation

This is an uncached-teacher-trajectory, fixed-age diagnostic, not a benchmark of
closed-loop adaptive inference. Correlation does not establish that the proxy
can safely trigger refreshes or improve final image quality. High pooled
correlation alone can come from cache age and diffusion stage. Adaptive V2 must
still be tested at equal compute budgets, on unseen prompts, and on its own
cached trajectory. A second interval sweep would test robustness beyond ages
1 and 2. Reported runtimes include expensive oracle instrumentation and must not
be presented as production inference latency. Saved images are full-DPM20
references, not cached-policy outputs. No thresholds are fitted here.
