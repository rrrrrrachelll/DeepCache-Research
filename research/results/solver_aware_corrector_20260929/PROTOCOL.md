# Solver-Aware Cache Corrector: Week-One Protocol

## Frozen question

Can a sub-million-parameter corrector predict the guided-noise error of a
branch-0 DeepCache reuse call on its actual closed-loop trajectory, and does
the previous DPM-Solver++ state add useful information beyond the current
latent, cached prediction, cache age, and timestep?

This week-one experiment is a feasibility gate, not a final quality claim.
The old one-time held-out prompts are not used.

## Fixed inference setup

- Stable Diffusion v1.5, fp16, 512x512, CFG 7.5.
- DPM-Solver++ order 2, midpoint, 20 steps, linspace timesteps, no Karras.
- DeepCache branch 0.
- Development refresh indices: `[0, 2, 5, 7, 10, 13, 16, 19]`.
- The cache trajectory, not the full shadow prediction, advances the scheduler.
- At reuse calls the full and cached UNets see the same latent, timestep, and
  prompt embeddings. The full call cannot read or update the real cache and
  cannot advance the scheduler.

## New prompt split

`prompt_set.py` deterministically defines 180 prompts in twelve categories.
Splits are prompt-disjoint: 120 train, 30 validation, 30 closed-loop
development. Training uses one seed; validation and development use two.
Previously consumed held-out prompts are excluded.

## Week-one stages and gates

1. Unit and real-model smoke checks.
2. Twenty-trajectory pilot collection and integrity report.
3. Round-0 MVP collection: 120 train and 60 validation trajectories.
4. Train C0 (no solver history), C1 (age/stage), C2 (one history), and C3
   (two histories); include shuffled-history control for the best history model.
5. Single-step gate: mean recovery >=30%, positive on >=9/12 reuse steps and
   >=8/12 prompt categories, history gain >=5 percentage points, and real
   history better than shuffled history.
6. Only if the single-step gate passes, run the 60-trajectory closed-loop
   development comparison. No old held-out data may be inspected.

## Integrity requirements

- Disabling the corrector must preserve the scheduled DeepCache result.
- Shadow full calls cannot mutate scheduler or cache state.
- Refresh calls have exactly zero full/cache error.
- CFG is combined exactly once before correction.
- Saved histories are scheduler-converted DPM-Solver++ data predictions.
- Every run records source hashes, prompt hashes, versions, GPU, configuration,
  completed cases, timing, and peak allocated memory.

## Stop rule

Do not enlarge the model or dataset if the one-history/two-history correctors
fail the predeclared single-step gate. A failure ends this branch.
