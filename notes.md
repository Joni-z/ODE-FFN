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
