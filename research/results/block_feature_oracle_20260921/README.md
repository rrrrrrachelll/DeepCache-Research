# Block Feature Error Profiling + Activation Replacement Oracle

This development experiment profiles where balanced8-A DeepCache activations diverge from a full SD1.5 / DPM-Solver++ 20-step pass and tests whether restoring each block activation causally reduces guided-noise error.

For every denoising step, the full UNet is evaluated on the scheduler's current latent and its block activations are recorded. A scheduled DeepCache pass is evaluated on the same latent. At every age-2 reuse step (`4, 9, 12, 15, 18`), each candidate activation is replaced separately with its full-pass value and the remaining UNet suffix is recomputed. Probe outputs do not advance the scheduler or alter persistent cache state.

The run uses five diverse prompts from the original analysis split with seeds 0 and 1. Held-out prompts remain untouched. The main outputs are block feature relative-L1/cosine distance and activation-replacement recovery of guided-noise error.

```bash
HF_HUB_OFFLINE=1 /root/miniconda3/envs/deepcache/bin/python -B research/results/block_feature_oracle_20260921/run_experiment.py
/root/miniconda3/envs/deepcache/bin/python -B research/results/block_feature_oracle_20260921/analyze_results.py
```

This is a teacher-forced local oracle. It is intended to choose candidate blocks for selective recomputation, residual correction, or trajectory distillation; a later closed-loop image experiment is required for final quality and speed claims.

Block names use DeepCache's deep-to-shallow logical indexing. In particular, `up_block_1` is `pipe.unet.up_blocks[2]` in physical module order.
