# Figure catalog

All five PNG/PDF figures use the full audited four-model cohort. No confidence
bars are shown: the plotted arms/checkpoints are fixed comparisons. Display
clamping does not affect stored raw predictions or sampler states.

## fixed-comparison

Purpose: show the opposite fitting/FID rankings without selecting one metric.
Data: comparison.csv; weighted/clean validation MSE and 1,024/2,048 FID.
Observation: ODE has lower measured FID while SwiGLU fits better. K2's fitting
and order gates fail. Caption must state sample counts and that lower is better.
Do not equate this small image pilot with a published FID-50K result.

## learning-curves

Purpose: distinguish endpoint arithmetic from a visibly different trajectory.
Data: all 44 audited validation checkpoint rows. Log vertical axes.
Observation: K1/K2 track closely and SwiGLU's fitting advantage persists through
much of the run. No best-checkpoint selection. Checkpoints are correlated
observations, not repetitions or uncertainty intervals.

## P-and-clocks

Purpose: show what the learned P/controller parameters do, rather than report
FID alone. Data: all 48 P records and 240 clock panel rows. Top: maximum singular
value for each of six matrices. Bottom: clock means averaged uniformly over
the five fixed 64-image panels, each containing 64 tokens/image. Clock axis is
its full allowed 0.2–1 range. All 64 singular/eigenvalue entries remain in CSV.
Observation: finite, non-flat P spectra and active clocks do not produce a K2
order win. These are local-generator diagnostics, not full-network stability.

## generation-cases

Purpose: make the magnitude of the FID difference visually assessable.
Data: same first eight registered Gaussian starts, all models, Heun32.
Observation: coarse layouts are similar and all images are blurred. The third
and sixth columns retain similar light/dark structures. No proof of improved
mode coverage or semantic detail. Examples were fixed by index, not selected
for appearance; all 1,024 raw outputs per arm remain private.

## denoising-case

Purpose: connect regression differences with a concrete held-out target.
Data: first validation image, all five predeclared noise levels, shared source,
all four final predictors; numeric values in first-case.csv. Target/noisy and
predicted images use the same display conversion.
Observation: coarse color/layout appears before fine details; at t = 0.9 K2's
case error exceeds Swi129's. This one case does not explain global FID or
replace the full validation table. No class label is used by the models.
