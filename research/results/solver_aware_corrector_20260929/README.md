# Solver-Aware Cache Corrector (SACC)

Run from the repository root with the established environment:

```bash
/root/miniconda3/envs/deepcache/bin/python -B -m unittest \
  research.results.solver_aware_corrector_20260929.test_sacc -v

HF_HUB_OFFLINE=1 /root/miniconda3/envs/deepcache/bin/python -B \
  research/results/solver_aware_corrector_20260929/collect_round0.py \
  --output-dir research/results/solver_aware_corrector_20260929/outputs/pilot \
  --split train --limit 20
```

The collector is resumable and never overwrites an existing case. See
`PROTOCOL.md` for frozen gates and interpretation boundaries.
