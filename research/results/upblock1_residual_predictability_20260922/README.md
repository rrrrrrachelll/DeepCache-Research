# up_block_1 Residual Predictability

This experiment tests whether the teacher residual at logical `up_block_1` can be predicted by deployable, prompt-generalizing channel models. It uses all ten analysis prompts and seeds 101/202, holds out both seeds of one prompt together, and never uses the consumed held-out split.

M1 predicts a step-specific channel mean residual. M2 fits a step-specific channel affine residual from cached activation. Both predictions are injected at `up_block_1`, followed by suffix recomputation. The teacher residual with the same suffix is the Oracle upper bound.

```bash
HF_HUB_OFFLINE=1 /root/miniconda3/envs/deepcache/bin/python -B research/results/upblock1_residual_predictability_20260922/run_experiment.py
/root/miniconda3/envs/deepcache/bin/python -B research/results/upblock1_residual_predictability_20260922/analyze_results.py
```
