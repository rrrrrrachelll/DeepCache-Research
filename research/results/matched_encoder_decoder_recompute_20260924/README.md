# Matched Encoder–Decoder Path Recompute

This teacher-forced oracle progressively recomputes matched encoder-to-decoder paths at balanced8-A age-2 steps 4 and 9. It tests whether refreshing the encoder skip sources consumed by the decoder restores guided semantic information that decoder-only recomputation could not recover.

The five paths are `decoder_only`, `down3_pair`, `down2_pair`, `down1_pair`, and exact `full_unet`. The experiment uses SD1.5, DPM-Solver++ 20 steps, branch 0, the 10 analysis prompts, and seeds 101 and 202. Previously consumed held-out prompts remain excluded.

## Completed results

Two complementary sweeps are complete. The contiguous encoder-to-output frontier produced no passing partial path. The stricter symmetric skip-matched sweep found that `matched_012` recovers 41.29% of guided-noise error and is positive for 10/10 prompt groups, but saves only 104.0 ms per probe versus a full UNet recomputation. `matched_0123` recovers 45.75% and saves only 35.1 ms. Cheaper `matched_0` and `matched_01` do not reliably recover error. Therefore symmetric skip consistency is necessary, but partial matched recomputation is not a useful speed-quality operating point on this model.

See `outputs/REPORT.md` for the contiguous sweep and `symmetric_outputs/REPORT.md` for the final symmetric sweep.
