# ODE mechanism investigation

Goal active: identify which ODE constraints help generation and why, using
falsifiable comparisons. Seed 0 only; native source and finalized reports fixed.
145 physical 0-3 forbidden; 4-7 only when empty. Never stop/share another task.

## Completed

- [x] Eight new 20k trajectories: free Taylor, true state flow, free midpoint
  stages and released stage operators, on Ring8/Product16.
- [x] Native-entry/continuation, initialization/RNG/gradient pairing, common
  banks, active-parameter interventions, matrix work and ordinary cost controls.
- [x] All 25 earlier resolution/symmetry cases and 78 scores independently checked.
- [x] Single-generator sharing primary evaluations: wins 8/8 banks on both tasks.
- [x] P0 full GPU refinement: 16 pooled / 128 bank scores independently checked.
- [x] P1 exact group operators, oracle equivariance and field-MSE identity.
- [x] Bind eight checkpoints and four fresh banks; CPU/CUDA real-entry checks.
- [x] Fresh documented CUDA witness, frozen5e78e6a, exit 0.
- [x] Complete 8 full frozen cases on idle 145 GPU 4/5; queue 2996899 terminal.
- [x] Independent64 bank / 16 pooled/32subset metrics, gap/balance/flux/path checks.
- [x] Five actual figures inspected; conclusions and decision rules recorded.

## Current findings

Shared P wins the original within-family comparisons. After symmetry
projection, Ring8 shared/stage SW2 reverses .072/.096 -> .024/.021;
Product16 retains .059/.068 -> .055/.063. Sharing lowers some symmetry error
but does not uniformly improve the preserved field. Ordinary controls remain
strong; no image promotion. Numerical replication error is at most 1.25e-16.

## P2: exact density-error diagnostic (implementation underway)

- [x] Derive score and oracle divergence from the pinned mixture law.
- [x] Verify exact continuity-residual identity and Gaussian rotation/radial
  counterexamples, with derivative checks against native log_density/velocity.
- [x] Register the eight frozen-cell/time-region measurements and evaluation
  stream before full measurement. Use exact traces in 2/16 dimensions.
- [ ] Only if a measured source term supports it, register a bounded common-
  source, two-resolution oracle-replacement intervention. No response search.
- [ ] Accept/reject localization from actual sampling; retain ordinary controls.

Canonical evidence:
- docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md
- sharing-symmetry/analysis-report.md within that study
- post-symmetry-decision.md within that study
Raw:/Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry
Remote:/data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116

P1errors and execution journal are preserved in
  docs/history/2026-11-16-ode-mechanism-p1-execution.md
The goal remains active: the selected symmetry does not explain the remaining
Product16sharing gap or establish an ODE-specific advantage over ordinary FFNs.

Previous goal turn: progress. It completed and independently verified all eight
P1cases, published five inspected figures and rejected the single selected-
symmetry account. Current turn derives and implements an exact along-trajectory
log-density diagnostic. Averages of source-residual norms remain secondary.

P2execution: first verify native oracle derivatives, continuity and Gaussian
counterexamples on CPU; then instrument the unchanged native Heun calls with
exact model Jacobians. Check density/Jacobian quadrature and endpoint refinement
before formal interpretations. Recheck idle145GPUs4-7; all are currently occupied.
No new training is needed. Preserve P1arrays and all finalized reports.

P2CPUauthor readiness passes all eight native models in both original/projected
forms, including exact native sampling identity and two actual CLIentries.
Evaluation stream2026112300is bound, with separate numerical pilot and four
256-source banks. The observer instruments unchanged native Heun calls. Next:
publish source, fresh documented CPUwitness, then monitored eight-case pilot.
