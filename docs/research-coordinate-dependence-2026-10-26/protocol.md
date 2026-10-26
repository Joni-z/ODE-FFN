# Coordinate-redraw diagnosis of Product16 nonlinear error

Registered before generating or evaluating these new probes. Raw-affine error explains only about4% of the direct ODE excess on old Product16 panels. That does not distinguish an incorrect nonlinear response along each relevant coordinate from dependence on other coordinates. Earlier off-diagonal Jacobian checks are local derivatives on different states; they do not quantify finite population error caused by cross-coordinate dependence.

## Population and fixed models

Use the original Product16 population and rotation from the frozen conservative-readout source. In latent coordinates, independent entries at fixed t have law

```text
q_i(t) = (1-t)*z_i + t*(sign_i*mu_i + .2*eps_i),
mu_i=geomspace(.75,2.5,16).
```

The oracle velocity in latent coordinate i depends only on q_i,t. This is a property of this explicitly constructed evaluation population; cached-bank training has a different, generally coupled conditional mean. The known rotation is used only for diagnostics, never supplied to training or architecture initialization.

Retain all seven final Product16 arms from conservative_readout_20261026: ODE K2/SwiGLU128/SwiGLU192, each direct/complete-gradient, plus direct SwiGLU384. Final step5000 only. Restore the actual Fourier buffers and checkpoint state; recheck all35 existing native float32 panels before promotion to float64 for the new experiment. Preserve the original model and target semantics. No inference-depth change, coefficient fitting, output correction or optimizer step.

## Common paired redraw design

Draw N=4096 independent row pairs of base Gaussian sources, signs and target Gaussian perturbations, using a dedicated NumPy generator seeded0. Reuse each row's standardized draws across times .1/.3/.5/.7/.9 and every model. This is a diagnostic sampling seed, not an additional training seed.

For each time, construct independent latent vectors A and B. For output coordinate i form H_i with A_i retained and every other coordinate taken from B. Evaluate the full native predictor at rotated A and H_i, rotate its velocity back, and retain output i. This keeps the exact population law at each evaluation. Save all standardized draws, latent inputs and all native/hybrid outputs.

Let eA_i and eH_i be the two errors against the same oracle value at A_i. For every row/output define

```text
T_i = (eA_i²+eH_i²)/2
C_i = eA_i*eH_i
D_i = (eA_i-eH_i)²/2
T_i = C_i+D_i.
```

Their expectations decompose population risk: C is the squared conditional-mean error E[(E[e_i|q_i])²]; D is E[Var(e_i|q_i)]. Because the oracle is unchanged by the redraw, D can also be computed from predictor differences without any oracle labels. D measures all dependence on the other coordinates, including additive leakage, not only higher-order interactions. Finite-sample C can be negative; do not clip it. Average over16 outputs, then the five fixed times, keeping each original row as the paired sampling unit.

## Primary question and uncertainty

For direct ODE versus each direct SwiGLU128/192/384, compute signed differences DeltaT, DeltaC and DeltaD, and the fraction DeltaD/DeltaT when DeltaT>0. All references must be retained. Use512 paired row-bootstrap resamples with seed0 for95% percentile intervals, keeping all outputs/times/models of a row together. These intervals concern Monte Carlo uncertainty for fixed learned functions, not variation over training runs.

- Dominant extra-coordinate explanation supported only if every reference's lower interval bound is at least .50.
- Rejected as the majority explanation only if every reference's upper bound is below .50.
- Otherwise report mixed/inconclusive; do not adjust probe count, coordinate grouping, time subset or threshold after outcomes.

Keep the three gradient-readout models and direct cost control visible. Their outcomes describe whether the mechanism changes with readout; they cannot replace the primary question or reopen the stopped potential architecture. No architectural training is selected solely by this experiment.

## Validation and resources

Before outcomes verify the identity and exact conditional decomposition on an exhaustive independent Rademacher product; include a known separable nonlinear error and an off-coordinate leakage error. Check latent/physical rotation round trips, oracle coordinate independence and all recorded native panel predictions. In float64 check complete-potential predictions against a separately implemented explicit-adjoint evaluator on a bounded subset of the actual new inputs.

Use one Torch CPU thread, chunks256 and bounded memory on Mac mini. Log actual process/time/memory after each model/time; poll the same process handle until terminal. Lenovo is currently offline, so no GPU reservation or remote job is claimed. Freeze source/protocol and bind all seven checkpoints, original source/banks/results and new probe arrays. Independent NumPy analysis must reconstruct every T/C/D row, all summaries, bootstrap intervals and the final decision from saved fields. Do not re-evaluate a live job on an observation timeout.
