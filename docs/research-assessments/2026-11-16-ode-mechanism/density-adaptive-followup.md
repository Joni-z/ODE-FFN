# Resolve the sensitive paths before interpreting density

The full Ring8 ordinary controls exposed resolution-sensitive paths that the
eight-source pilot missed. At 1024/2048 steps, Free ISRU has 58 original and
23 projected failing source observations; SwiGLU has 25 and 11, respectively,
each out of 1024. Formula reconstruction passes. The missing evidence is
convergence of the paths and their accumulated density ratios.

## Fixed numerical follow-up

Keep the eight checkpoints, four common 256-source banks, both fields,
float64, native Heun sampler and exact derivative observer. Never change
the models, source arrays or acceptance tolerances. Do not rank density
estimates while any required comparison remains unresolved.

For each model/field, retain every source and its 1024/2048 observations.
Only sources failing at least one existing gate are integrated again at
4096, then 8192, then at most 16384 steps. A source stops when its adjacent
resolution comparison passes all five gates. Selection depends only on
numerical errors, never the sign or size of the estimated density contrast.
The final file retains all 1024 sources, their bank/source identifiers,
selected resolutions, previous-resolution estimates and acceptance flags.
No difficult source is removed from averages.

Batching a subset is valid here because the eight frozen models and group
projections have already passed sample-separability checks. Each new subset
is also checked against the unchanged native sampler at that resolution.
Preserve all subset arrays and parent hashes. Independently reconstruct
the density formulas, source selection, merge and per-point gates before
using the merged arrays. A witness exercises two small resolutions and the
same merge, but cannot qualify population estimates.

The 16384-step cap is a stopping bound, not a relaxed acceptance rule. Keep
unresolved points and mark their entire model/field cell unresolved. This
follow-up does not alter the running 128-row producer or relabel its results
as converged.

## Statistical analysis after qualification

Compare stage-dependent minus shared-generator terminal log-density ratio
on common Gaussian sources for both tasks and both fields. Use all four
banks, with a stratified paired bootstrap over whole paths. Report means,
SD, Monte Carlo SE, 95% intervals, and the changes from each point's previous
resolution. These are fixed-checkpoint evaluation uncertainties.

For the four predeclared contrasts, also report Bonferroni 98.75% marginal
intervals as an approximate familywise 95% interval family. Do not use time
bins or resolutions as extra independent observations. Do not clip negative
finite-sample mean log-density ratios. Report tail contributions without
trimming them.

As a paired metric check, also replay the pinned native SW2 formula on the
same 1024 endpoints, prebound target banks and 256 directions. Report pooled
and four bank scores, plus previous-resolution scores. This smaller cohort
has a higher empirical sampling floor; it supplements the completed large
sampling comparison and does not replace it. Independent NumPy and native
Torch replay must agree before using these descriptive scores.

Only bank 0 may be used to inspect time-bin profiles and select a single
candidate time window. Banks 1-3 remain hidden at the time-bin level until
an oracle-replacement prediction is registered. If the resolved density
contrast disagrees with the known SW2 comparison, reject this proposed
explanation of the SW2 advantage rather than choosing a favorable window.

## Execution boundary

Mac mini CPU may run freely with one thread per worker. On 145, only a
freshly idle physical GPU 4-7 may be selected; retain the existing exclusive
UUID guard and stop only our process if another compute PID appears. Use
the warm environments and validated coordinate CPU / batched CUDA backends.
Publish and verify the real invocation before starting this follow-up.
