# Current project status

## Hidden changes reach the terminal distribution in a bounded pilot

The fixed-first-layer pilot is complete: six shared/PAL/ALT models and four
ordinary controls,64 common sources per checkpoint, inner256/512 and native
Heun128/129. Of12 focal model/grid comparisons,9 pass every numerical gate;
all9 support the registered directional prediction. Their tangential
reproduction fractions G range from0.926 to0.996. This measures reproduction
of the full refinement's empirical distribution change, not a target-quality
improvement. Three Ring8 comparisons retain failed256-step snapshot gates;
all512-step snapshots pass. No case or arm is removed.

[Complete analysis and figures](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-results/analysis-report.md),
[all numerical data](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-results/summary.json),
[source and protected-file audit](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-audit.json).

Target SW2 changes are small and do not all improve. The64-source pilot cannot
rank methods against earlier32768-source studies. It does not select a new FFN.

## The next mechanism control

Normal and tangential spaces have dimensions1 and63. Directional dominance
alone may reflect that difference. An exact audit of all2880 old local states
shows normal error is enriched2.158--16.489 times relative to the1/64 isotropic
reference. Thus energy-only is insufficient, but the potential normal is not
unimportant. Compare actual corrections with complete, equal-dimension
orthonormal-frame controls before promoting this as special ODE sensitivity.
[Derivation, measured enrichment and falsification](docs/research-assessments/2026-11-16-ode-mechanism/direction-dimension-audit.md).

Also register a512/1024 numerical extension before resolving the three remaining
pilot gates. Preserve the original256 failures and all fixed samplers/sources.
The dimension-matched control is registered and all36 complete frames are
bound before new outputs. Both first-layer author witnesses and independent
verifiers exit0, retaining all64 directions and80 inputs in all six models.
Fresh frozen execution is next. No larger sampling run or architecture change.

## Earlier evidence and image benchmark

The complete one-layer output study supports a directional local response on
all six models. [Results](docs/research-assessments/2026-11-16-ode-mechanism/hidden-readout-results/analysis-report.md).
The all-layer native reference qualifies256 on Product16 but fails late layers
on Ring8; the restricted first-layer pilot does not overturn that result.
[Reference qualification](docs/research-assessments/2026-11-16-ode-mechanism/hidden-surrogate-results/analysis-report.md).

CelebA128 at160 epochs remains native24.304/K1 27.789/K8 28.083 FID50k.
[Matched image table](docs/celeba128-wide-20261114/continuation-160/matched-results.md).
Reports1/2 are final; all97 protected report files,11 Report2 bindings and five
native image source files remain unchanged.

## Execution and resources

All pilot workers, independent verifiers and monitor have actual terminal
exit0. Runtime freeze9d46b2d, original sessions2572/38657; Elapsed time per task
approximately312--317 seconds using two one-thread Mac workers. Warm environment
04f11f75 is unchanged. Raw outputs and monitoring:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/`.
145 physical4-7 were occupied;0-3 prohibited. No remote job was stopped/shared.
