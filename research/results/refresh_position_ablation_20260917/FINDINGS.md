# Findings

## Main result

Refresh position has a large effect even when the compute budget and the maximum
anchor span are held constant. Restoring the early step-2 refresh improves SSIM
by 0.10542 on average (prompt-grouped 95% CI [0.06989, 0.14802]) relative to
`drop_early8`. The benefit is positive for all 10 development prompts.

The middle step-10 refresh has a much smaller and uncertain effect: mean SSIM
benefit 0.00487, 95% CI [-0.00077, 0.00995], positive for 7 of 10 prompts. The
late step-17 refresh has a small but consistent effect: mean SSIM benefit
0.00941, 95% CI [0.00626, 0.01249], positive for all 10 prompts.

The step-2 benefit is larger than the step-10 benefit by 0.10055 SSIM and larger
than the step-17 benefit by 0.09601 SSIM; both paired prompt-bootstrap intervals
exclude zero. Early refresh is therefore substantially more valuable to final
DPM20 fidelity than either tested middle or late refresh.

## Interpretation

The failed stage-aware schedule removed the uniform schedule’s step-2 refresh and
spent more full calls late. This ablation isolates that change and shows why the
policy degraded: an early approximation error changes the subsequent diffusion
trajectory, so its final-image cost can be large even when the local guided-noise
error measured at later steps is larger. A proxy based only on current local
error or cache age will miss this propagation effect.

The next controller should keep the early refresh window as a hard scheduling
constraint. Proxy-controlled or stage-aware reallocation should operate on the
middle and late refreshes first. A focused early-window sweep that moves the
second full call among steps 1, 2, 3, and 4 at a fixed total call budget is the
most direct next experiment before designing a more complex learned proxy.

## Data note

The safety checker replaced `objects_202/drop_early8` with a black image during
the first pass. That exact prompt, seed, latent, and schedule were regenerated
with image replacement disabled; its final `nsfw_flag` is null. All other rows
retain the normal safety-checker result. This repaired row is valid for fidelity
metrics, while its timing and peak-memory values should not be used for a
single-case safety-checker comparison.

Held-out prompts were not used.
