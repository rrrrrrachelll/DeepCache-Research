# Next Experiment: Current-Step Spatial Feature Conditioned Residual Predictability

## Evidence motivating the experiment

The completed channel-statistics experiment preserves a large teacher oracle at logical `up_block_1` (67.35% guided-error recovery), but channel mean and per-channel affine predictors recover only 2.61% and 4.21%. The failure is concentrated at steps 4 and 9, where the teacher oracle is strongest. This supports the hypothesis that the missing residual is spatially and content conditioned.

The current result establishes insufficiency of the tested channel-statistics models. It does not prove that every higher-capacity cached-only predictor is impossible, so a cached-only low-rank spatial model is retained as a capacity control.

## Causal, inference-available inputs

For each age-2 probe step, use only tensors available to an actual cached pass:

- `C`: cached logical `up_block_1` activation, 640 channels.
- `Delta_conv`: current `conv_in` activation minus its last-refresh anchor. This supplies current latent spatial motion but is identical across CFG branches.
- `Delta_shallow`: cached-baseline logical `up/attentions/layer_0` output minus its last-refresh anchor. This layer is recomputed by branch-0 DeepCache and supplies current spatial and text/CFG-conditioned information.
- Step or log-SNR, implemented by fitting separate parameters for each of steps 4, 9, 12, 15, and 18.

Teacher activations may be used only as regression targets and oracle controls. They must never enter predictor inputs. `Delta_shallow` must be captured from the cached baseline pass, not the full teacher pass.

Because the target has 640 channels while the current shallow signals have fewer channels, a per-channel depthwise model is not dimensionally sufficient. Use a low-rank 1x1 spatial linear map with fixed rank 16 and ridge regularization. Fit it from prompt-grouped sufficient statistics and evaluate by leave-one-prompt-out.

## Models

- `S0_channel_affine`: existing M2 result, reused as the lower baseline.
- `S1_cached_lowrank`: low-rank map from `C` only. This separates model-capacity gains from gains due to current-step information.
- `S2_conv_delta`: low-rank map from `[C, Delta_conv]`.
- `S3_shallow_delta`: low-rank map from `[C, Delta_shallow]`.
- `S4_combined`: low-rank map from `[C, Delta_conv, Delta_shallow]`.
- `shuffled_current_control`: evaluate S4 after permuting current-step deltas across prompt groups. Recovery should collapse if the spatial signal is causal rather than a larger-model artifact.
- `teacher_oracle`: true target activation followed by the same full suffix recomputation.

All predicted activations are followed by full suffix recomputation, matching the established propagation requirement.

## Protocol

- SD1.5, DPM-Solver++ 20 steps, CFG 7.5.
- DeepCache branch 0 with balanced8-A refreshes `[0, 2, 5, 7, 10, 13, 16, 19]`.
- Probe steps `[4, 9, 12, 15, 18]`.
- Existing 10 analysis prompts x seeds 101 and 202.
- Prompt-grouped LOPO; both seeds of one prompt are held out together.
- Previously consumed held-out prompts remain unused.
- Main comparisons are paired on identical initial latents and the same teacher-forced trajectory.

Record residual relative-L1, corrected-feature relative-L1, guided-noise error, recovery fraction, oracle-gain fraction, positive prompt groups, and predictor plus suffix GPU time. Report steps 4 and 9 separately as primary diagnostics.

## Predeclared decision

Proceed to a closed-loop image experiment only if one current-conditioned model satisfies all of:

- mean guided-error recovery at least 20%;
- positive mean recovery on at least 7 of 10 prompt groups;
- non-negative recovery at both steps 4 and 9;
- clear improvement over `S1_cached_lowrank` and the shuffled-current control;
- estimated end-to-end runtime remains faster than DPM20 and PNDM50+DeepCache when measured in one matched run.

If no current-conditioned model passes, stop the logical `up_block_1` residual-predictor route and move to local selective recomputation or a redesigned cache representation. Do not increase predictor depth without evidence that the new current-step inputs contain sufficient signal.
