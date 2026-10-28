# Does the 20k image lead survive fixed local clocks?

Owner: Zhizhe Zhang. Register before new intervention outputs.

## Question and fixed cohort

The completed budget study gives native K1 FID 173.43691 at 20k, versus 195.31010 for Swi86 and 185.88760 for SiLU131. K1 meets the predeclared follow-up margin; K2 does not. At 5k, simultaneous clock collapse retained the measured image lead, but those weights and clock distributions differ. Test the same finite intervention at the actual 20k checkpoints before designing another training family.

This is two new frozen interventions, K1 constant and K2 constant, plus all seven reused native 20k endpoints. No training, seed, architecture, loss, coefficient, solver, matrix scale, calibration search or extra endpoint is introduced. The older failed settings and submitted Report1 remain unchanged. K2 is an order/context control, not a newly accepted candidate.

Budget parent: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget-v2`, seal `ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b`.
Reuse the unchanged native banks/construction from cycle17, Heun128 from cycle18, and literal clock/calibration functions from sealed cycle19-v2. Bind all source and parent hashes. Restore each actual `step_0020000.pt`, including nonpersistent buffers. Never substitute a 5k or 10k model.

## One calibration and one intervention per model

Run exactly the first 64 existing validation inputs and their saved times through each unchanged native model. At each of its six ODELayer inputs, record native scalar durations across 64 images × 64 tokens. Average in float64 and store one float32 constant. Use no clean target, label information, FID, time bins or alternate calibration.

Replace all six local durations simultaneously by those constants. Preserve P, SiLU, attention, external time/AdaLN conditioning, readout, learned parameters and buffers. Recompute all downstream states using the original float32 recursive polynomial. The calibration is exposed validation data; no fresh holdout claim follows.

For a column state h and clock s(h), the local maps and their complete Jacobians are:

```text
K1: F(h) = h + s P h
    J(h) = I + s P + (P h) grad(s)^T

K2: F(h) = h + s P h + 0.5 s^2 P^2 h
    J(h) = I + s P + 0.5 s^2 P^2
           + (P h + s P^2 h) grad(s)^T
```

At constant c, the local Jacobian is the fixed polynomial matrix. The two-solve FFN is `M2 SiLU(M1 h)`. Collapsing clocks changes both polynomial coefficients and their input derivatives; it does not isolate the rank-one derivative term. Image and external-time effects carried through h are also not separated. These identities do not order FID or fitting error.

## Unchanged evaluation and declared decisions

Use all 1,024 existing validation pairs for weighted and clean MSE; all five fixed 64-pair time panels; all 1,024 common Gaussian starts with the exact CPU Heun128 update and denominator cap .05. Keep the same first eight generated starts for examples. No state clipping or sampler change. Reuse all seven audited native 20k sample/feature banks; extract only the two new Inception feature banks. FID remains 1,024 generated/2,048 validation references, identical pinned weights, MPS preprocessing and uint8 conversion. Report all nine endpoints, raw out-of-range fractions, feature-mean/covariance terms and native-to-constant deltas.

The primary question is K1 under this registered simultaneous collapse. Retain its declared Swi86/SiLU131 references and K2's Swi129/SiLU193 references. For each arm report separately:

1. Strict FID ordering: constant FID below both declared ordinary references.
2. Retained allocation margin: FID at most .95 times each reference, with weighted and clean MSE each at most 1.05 times the best corresponding reference.
3. Continuous fitting/FID changes from the native model and how much of its ordinary-control gap remains.

Retained ordering shows that the measured input variation is not necessary for that ordering at these weights/constants. A lost ordering shows sensitivity to this intervention, not that every constant model must fail or that a retrained fixed-clock model cannot work. Losing only the 5% margin is not the same as losing the ordering. K2's old failed margin stays failed regardless of a frozen change. No outcome establishes an ODE-specific causal mechanism or a new trained method.

Stop after the two specified interventions. No layer, temperature, clock-range, time-only-clock, calibration, source, solver or training rescue is selected from the outcomes.

## Readiness, resources and verification

Mac mini only; unchanged warm environments image-training@475428fb and image-metrics@dc729a50. No package installation or environment rebuild. Native source/construction and feature kernels were independently replayed in the sealed budget study. New documented preflight must be executed once by a fresh agent before formal generation.

Preflight checks full native 1,024-pair prediction replay at 20k, both actual checkpoint steps, unchanged parameters/buffers, all twelve calibration arrays/constants and constant-polynomial fixtures. Promoted whole-network literal/folded witnesses at the five fixed times require max absolute difference <1e-9; this is algebra-only, with the existing time-feature dtype lift. Formal predictors stay float32. Preserve failures and do not relax numerical tolerances.

Sampling uses at most two one-thread CPU children, 1,200 seconds per child and 1,300 seconds for the phase. Independent sampling audit ceiling 1,200 seconds; feature and feature-audit ceilings 600 seconds each. Monitor PID/create-time and primed five-second CPU/RSS; no GPU reservation for sampling. Run once, with no automatic retry. Expected additional storage below 300 MB. Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clocks20k`.

After generation, independently reproduce native calibration via explicit linear/SiLU operations, both new predictions, ten time panels and both full generated banks bitwise with a separate literal-clock/manual-Heun expression. Verify unchanged parameters/buffers and parent/source hashes before/after. Fresh feature audit reproduces all 2,048 new and 2,048 reference rows bitwise; independent SciPy gesvd recomputes all nine FIDs within 1e-8. Seven cached native feature banks retain their complete budget audit and seal. Archive actual terminal outcomes, all scores and all cases before drawing conclusions.
