# Does the hidden directional response reach the generated distribution?

Registered before these sampling interventions. Previous interpolation-state
qualification fails globally at256on Ring8late layers; that result is retained.
This study changes only the first chronological local layer, blocks.0.ffn.ode1.
That layer passes256in every fixed shared/PAL/ALTmodel. It is chosen by index,
never by a sampling score. Other five local layers remain native4throughout.

## Hypothesis and exact arms

For the same frozen checkpoint and same Gaussian sources, replace the first
layer on every field evaluation by identity/native4, full refined endpoint,
normal-corrected endpoint, tangential-corrected endpoint or exact-potential
ray. Definitions are identical to hidden-readout-protocol.md, now recalculated
at each actual current input rather than reusing interpolation-state outputs.
The prediction is that tangential correction reproduces more of the full
refinement's final-distribution change than normal or exact-potential matching.
The full refinement is not assumed to improve agreement with the target.

Use all six20kseed0shared/PAL/ALTmodels. Include the four fixed ordinary
SwiGLU/FreeISRUcheckpoints as unmodified controls, two per task. Preserve the
native Heunsampler, time embedding and weights. FP32anchors are checked before
conversion toFP64for these numerical comparisons. Do not compare new64source
scores with historical32768source scores.

## Fixed pilot and numerical checks

Use the first eight sources and targets from each of the original eight
selection banks:64ordered sources/model. No fresh randomness. Use native
Heun128and129as two distinct fixed grids; the second avoids relying only on
dyadic agreement. They do not establish continuous outer-flow convergence.
For each intervention run both inner256and512. Baseline is computed once
per grid and must replay exactly after hook removal. Every arm must preserve
the original parameter/buffer hashes. No new training or package installation.

Retain all endpoints and first-layer snapshots at field-call indices
0,64,128,192,and2*N-1on every arm/inner/grid. For each captured batch, solve
the exact piecewise hidden field with both original DOP853tolerances and
qualification gates. Keep every point. At those states compare the runtime
candidate with its DOP-based counterpart using the original local and full
output limits:1e-9+1e-3times local refinement and1e-10+1e-3times full-output
refinement norm. Any unresolved reference remains visible.

Before interpreting a256-versus512sampling comparison, require every arm's
endpoint change to be <=1e-7+0.01times its512-versus-native endpoint displacement,
point by point. For each bank and pooled sample, require the absolute change
in the distribution-reproduction fraction below to be <=0.005. Retain both
levels and every numerical failure; do not loosen limits or select sources.
These are numerical approximation gates, not performance-effect margins.

## Distribution measurements and falsification

For each bank and pooled64sources, project generated endpoints on the original
256fixed directions and sort along each direction. Let Q_cbe this empirical
quantile array for arm c. Define

```text
G_c = 1 - ||Q_c-Q_full||² / ||Q_native-Q_full||².
```

Unlike the earlier source-coupled endpoint proxy, this compares empirical
distributions and discards which individual noise produced each endpoint.
It remains a finite-direction empirical distribution diagnostic. It is not a
percent improvement in targetSW2and not a proof of population-law equivalence.
Zero denominator or a reference shift below the numerical endpoint scale is
unresolved. Require SW2(native,full512)>1e-7+100*SW2(full256,full512)for every
bank and pooled sample; this resolves the measured distribution change at
one percent of its observed numerical-shift scale. Primary prediction:
G_T>0,G_T>G_N,G_T>G_rayfor every six-model case
at both fixed outer grids after numerical gates pass. A reversal rejects that
general six-case claim; unresolved numerics do not count as support.

Also report paired bank and pooledSW2to the common targets for all arms and
ordinary controls, plus within-source endpoint displacements as secondary
quantities. TargetSW2and reproductionGanswer different questions. No training
population inference is attempted. The64source pilot is descriptive; a later
full-bank experiment requires a separate decision after all outcomes are known.

## Execution and stopping

Reuse warm MacCPU04f11f75at two one-thread workers, one dataset each. First
check all six first-layer hooks against their saved80input candidates and
full outputs, then a fresh documented actual-sampler witness. Formal pilot
must retain every case/arm/grid/level and source. Native-math or binding failure
stops that entry and is logged; no observation timeout triggers a restart.
No145card is used while occupied; physical0-3remain prohibited. No image
architecture or finalized report is changed or promoted from this pilot.
