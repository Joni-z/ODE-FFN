# P2 numerical qualification journal

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

P2pilot complete: frozen8691431, queue32943and all8workers terminal. All48
resolution rows and16field cells independently verify within2.94e-14. Only
6/16density cells satisfy all512-step numerical gates. Ten require the
registered conditional1024/2048refinement; no density ranking is allowed yet.
Native full-sampler finite-difference log determinants pass8cases, max5.25e-8.

# ODE mechanism: current evidence

## Structural result

At the tested budget, state feedback and sharing P are the positive
within-family interventions. Free Taylor coefficients/free midpoint stages
match prescribed coefficients in sampling; the prescribed midpoint retains
order 2 while free stages approach order 1. Higher numerical order is a real
property, but does not automatically improve generation.

Sharing primary iid SW2: Ring8 .091/.111 and Product16 .064/.073
(shared/released), wins8/8paired banks each. Released stages add73,728params
at equal dense work; their learned actions and ordering effects are active.
Product16field fitting improves while sampling worsens. Full outer refinement
preserves the gap; hidden refinement weakens Ring8bank support to5/8.
Ordinary Product16controls remain stronger and cheaper.

## P1completed

New group-stratified arrays, stream2026112200, four4096-base-point banks:
Ring8original/projected shared .072/.024, released .096/.021;
Product16shared .059/.055, released .068/.063.
All four Ring8banks reverse; all four Product16banks retain sharing's lead.
Ring8FreeISRUprojected .019; Product16SwiGLU .040/.039.

Exact orbit-MSE split shows reduced breaking error but worse preserved error
under Ring8 sharing. Product16stage projected MSEis lower, generation worse.
The simple selected-symmetry explanation fails. Eight-bin projected masses
are exactly balanced for all models, yet SW2 still differs.

All 8 frozen cases exit 0, queue 2996899 terminal, physical145GPU4/5only.
Fresh documented CUDA witness2988104passed8 references and 2 entry checks.
Independent NumPy verification at c02da14 checks64 bank / 16 pooled/32subset
scores, decompositions, source bindings, gaps, counts, flux matrices and paths;
maximum disagreement 1.25e-16. The subset128->256SW2change<=5.44e-4,
but individual endpoint coupling RMS reaches .065. Interpret distributions.

Producer source 5e78e6a. Raw root:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry
Remote root:
/data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116
All datasets/checkpoints/raw samples remain outside Git. Final reports fixed.

## Next mathematical question

Exact density difference obeys
  partial_t(rho-p)+div(v * (rho-p)) = -div(p*(v-v_oracle)).
The normalized density-error source is div(e)+e dot score(p).
This retains divergence and score alignment that average field MSE discards.
It is a candidate diagnostic, not yet an observed explanation; transport may
amplify/cancel this source. See post-symmetry-decision.md for bounded checks.
No new training, response search or image run is launched from this proposal.

P1execution/errors preserved in docs/history/2026-11-16-ode-mechanism-p1-execution.md.
Warm145Python:/data2/zz5070/miniconda3/envs/py312/bin/python, Torch2.8.0+cu128.
Warm localTorch:/Users/mr.z/ode-workspace-prep/.venv-test/bin/python.

P2exact log ratio: along X'=v, d log(rho/p)/dt = -[div(e)+e dot score(p)].
It can be checked independently with the oracle log density and integral of
model divergence. The oracle derivative/continuity errors are below1.03e-12.
Eight real model observers pass native endpoint identity; coarse16-step density
chain residuals are intentionally not performance results. New sources and
derivative anchors are bound before the128/256/512numerical pilot.

P2pilot observations are produced and independently verified:16cells,48rows,
only6cells qualify numerically at512. The other10require the registered
1024/2048refinement. Local residual RMSalone does not track the known
Product16sharing advantage: shared2.116versusreleased1.995, though sharing
has lower prior samplingSW2. This is a field proxy, not a path-KLresult.
The exact signed along-path object remains the intended diagnostic.

A documentation-refresh substring mismatch stopped before the STATUS edit;
the corrected anchor preserves the completed evidence and source.
