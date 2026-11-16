# Current ODE mechanism evidence

## Completed distribution pilot

Only blocks.0.ffn.ode1 is changed dynamically; other five local layers and the
native sampler stay fixed. Six focal shared/PAL/ALT checkpoints plus four
ordinary controls use64 common sources each. Inner256/512 and Heun128/129,
all116 arms, five captured batches per focal arm and two DOP references retained.
Both native and independent exits0; all reference checks pass.9/12 focal/grid
comparisons qualify and support tangential reproduction G=.926--.996. Three
Ring8 comparisons fail256 snapshot output gates; all512 snapshots pass.
Target SW2 changes are small, sometimes worse. This is not architecture selection.

The metric compares sorted projected terminal samples with the full-refinement
arm, not corresponding-source endpoint vectors. It measures reproduction of a
change, not target-quality improvement. Ordinary controls use identical sources
and targets; no comparison with earlier32768-source values is made.

## Dimension changes the interpretation

The potential normal is1D and tangent space63D. For an independent uniform
unit normal r, E[||rr^T delta||²]=||delta||²/64. A complete orthonormal frame
has that average exactly. Audit on all2880 original local states reconstructs
normal shares and the frame identity(max error4.34e-19). Normal error is
2.158--16.489 times the geometric reference, despite its smaller absolute share.
Energy-only does not explain the whole effect; energy-normal structure is not
therefore irrelevant. The next finite equal-dimension controls are defined in
 direction-dimension-audit.md and have not run.

## Canonical evidence

- hidden-sampling-results/: complete pilot, strict statistics, figures, raw bindings.
- hidden-sampling-audit.json: source/runtime/97report/11final/five native-file checks.
- direction-dimension-audit.md/json and audit_hidden_dimension.py: exact geometry.
- hidden-surrogate-results/: preserved failed all-layer256 qualification.
- hidden-readout-results/ and hidden-energy-results/: complete earlier interventions.

All above are under docs/research-assessments/2026-11-16-ode-mechanism/ except
scripts under studies/muon_pd_20261102/target_geometry_20261116/.
Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/.
Native image trainer and Reports1/2 stay fixed. No GPU or package changes.
Next: dimension null and unchanged-gate512/1024 extension, before larger sampling.

## Current equal-dimension control

hidden-null-protocol.md fixes both hypotheses and12-contrast-adjusted bank
intervals. All36 frames were prepared with evaluation RNG0 and bound in
hidden-null-frames.json before any downstream observation. Both first-layer
author native and independent verifier exits0; candidate reconstruction is
exact. Full all-layer results remain pending; no hypothesis outcome is promoted
from the author witness. Same warm04f11f75, noGPU or package changes.
