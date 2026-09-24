# Early-Step Decoder Recompute Frontier Sweep

This teacher-forced oracle measures how much real U-Net computation must be restored to recover branch-0 DeepCache errors at early balanced8-A age-2 steps 4 and 9. Frontiers progressively recompute from logical `up_block_0`, `up_block_1`, `up_block_2`, `up_block_3`, or the mid block through the output; a full-UNet pass is the exact upper-bound control.

The experiment uses SD1.5, DPM-Solver++ 20 steps, 10 analysis prompts, seeds 101 and 202, and no consumed held-out prompts. Probe passes share the same latent and do not advance the scheduler or mutate persistent cache state.
