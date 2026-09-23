# Current-Step Spatial Feature Conditioned Residual Predictability

This experiment tests whether spatial signals that are genuinely available during a branch-0 cached pass can predict the logical `up_block_1` teacher residual. It compares cached-only, current `conv_in` delta, current recomputed shallow delta, their combination, a spatial-shuffle control, and the teacher oracle. Every corrected activation is propagated through the full suffix.

The protocol uses SD1.5, DPM-Solver++ 20 steps, balanced8-A, the 10 analysis prompts with seeds 101 and 202, and prompt-grouped leave-one-prompt-out validation. Consumed held-out prompts are excluded.
