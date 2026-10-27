# Cycle14 figure catalog

All four PNG/PDF pairs were visually inspected. They use audited results from every registered arm/law. No significance markers or training-variation error bars are shown; this is a fixed architecture screen. Shared banks, layers and checkpoints are dependent measurements. The plots do not turn the ten new trajectories or reused controls into replicate treatments.

## Fixed comparison

- Files: [PNG](results/figures/fixed-comparison.png), [PDF](results/figures/fixed-comparison.pdf).
- Purpose: inspect practical quality across both laws and the complete ordinary frontier.
- Data: [comparison.csv](results/comparison.csv), all seven model families on both laws.
- Variables: tangent field MSE, tangent-output Jacobian error and tangent SW2; bars start at zero. Field averages use 20,000 states, derivative errors five times/64 anchors, and SW2 the complete 2,048-row sample banks.
- Caption requirements: identify reused original/Swi192 and fresh Swi258; tangent-output derivatives retain every input direction. The main report/table also retains full-space metrics and fixed gates.
- Notice: C modestly improves thin data but loses to Swi192 on thick field error and sample distance. Wider Swi258 is not the stronger fitted thick reference.
- Decision: keep both ordinary references and stop the failed candidate; no model choice based solely on compute-matched width.
- Checks: all registered families appear, axes/units are explicit, no hidden discarded outcome.

## Learning curves

- Files: [PNG](results/figures/learning-curves.png), [PDF](results/figures/learning-curves.pdf).
- Purpose: show the saved learning trajectories and whether a final result hides a different recorded history.
- Data: [curves.csv](results/curves.csv), updates 0, 1,000, 2,500 and 5,000 for every model/law.
- Variables: tangent field MSE on a logarithmic y-axis; these are checkpoint observations, not confidence bands or independent repeats.
- Caption requirements: describe cold clean-head behavior and the fixed final endpoint. No interpolation is an observed optimizer step.
- Notice: C/free/affine trajectories stay close; Swi192 finishes below the added-coefficient arms on thick data.
- Decision: no longer-training or intermediate-checkpoint rescue is selected.
- Checks: legend includes all models, log scale is labeled and all saved updates are shown.

## P and coefficient histories

- Files: [PNG](results/figures/P-coefficient-histories.png), [PDF](results/figures/P-coefficient-histories.pdf).
- Purpose: check whether the additional direction remains inactive and record what P learns.
- Data: [P-history.csv](results/P-history.csv), all 192 new coefficient-layer/checkpoint records; full singular arrays are in [coefficient-singular-values.csv](results/coefficient-singular-values.csv).
- Variables: P largest singular value, C Frobenius norm and P/C commutator norm. Lines average the six solves; shading is their min/max range, not uncertainty. Direct B's C is derived as B−P².
- Caption requirements: all matrices are hidden operators; nonzero activity or commutators are not complete-model stability or quality certificates. Reference B is not the actual input-dependent coefficient for affine/free controls.
- Notice: C becomes active in every layer and the commutator grows, while the quality guards still fail.
- Decision: do not explain failure by a permanently dead branch or infer benefit from an expanded conditional family.
- Checks: separate law panels, native versus promoted diagnostic definitions preserved, no averaged spectra substituted for raw histories.

## Generation cases

- Files: [PNG](results/figures/generation-cases.png), [PDF](results/figures/generation-cases.pdf).
- Purpose: inspect whether lower field/SW2 values correspond to recovery of the target's visible modes.
- Data: saved native target/generated arrays, all 2,048 rows for each target/model, projected by the unchanged saved basis.
- Variables: two tangent coordinates, one panel per family/law plus target; equal limits/aspect ratio.
- Caption requirements: this projection hides normal-space spread; it is not FID, a mode-coverage estimator or an independent sample set. Dots are the full bank, not a selected subset.
- Notice: coefficient variants retain a largely continuous annulus rather than the target's eight separated clusters.
- Decision: their modest thin metric gain does not establish successful mode formation or a useful generative method.
- Checks: every registered model appears, target is shown on each law, no selected favorable examples.
