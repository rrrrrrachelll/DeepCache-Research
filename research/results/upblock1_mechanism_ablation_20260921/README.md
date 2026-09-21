# up_block_1 Mechanism Ablation

This experiment follows the block activation oracle by separating the two operations that produced its strongest actionable result at logical `up_block_1` (`pipe.unet.up_blocks[2]`): teacher activation replacement and recomputation of the remaining UNet suffix.

At balanced8-A age-2 reuse steps, it evaluates `activation_only`, `suffix_only`, and `activation_plus_suffix` on the same full DPM-Solver++ trajectory. It uses the same five development prompts and two seeds as the block feature oracle. Consumed held-out prompts are not used.

```bash
HF_HUB_OFFLINE=1 /root/miniconda3/envs/deepcache/bin/python -B research/results/upblock1_mechanism_ablation_20260921/run_experiment.py
/root/miniconda3/envs/deepcache/bin/python -B research/results/upblock1_mechanism_ablation_20260921/analyze_results.py
```
