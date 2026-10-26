# Review: recompute the original clock inside one solve?

**Decision: GO for one bounded, preregistered falsification screen, conditional on mathematical and implementation checks; no architecture is selected by this review.** This asks a previously untested question about the professor block, with a direct cached-clock control. It is not a continuation of the rejected forced-SwiGLU family. The evidence does not predict success, identify clock freezing as the existing failure's cause, or support repeated rescue if the screen fails.

## What changes, and what was already tested

The professor's scalar clock, and its later vector extension, are legitimate input-conditioned linear solves. [OriginalODELayer](../../synthetic/models.py) computes the clock once before its Taylor recurrence. Writing `D(h)=diag(s(h))`, the frozen second-order map is

```text
f = D(h) P h
frozen_K2(h) = h + f + .5 D(h) P f.
```

The proposed field instead recomputes its clock from the evolving state: `f(z)=D(z)Pz`. One explicit midpoint step of duration one is

```text
m = h + .5 f(h)
state_midpoint(h) = h + D(m) P m.
```

Replacing D(m) by the original D(h) gives exactly frozen_K2. The cached clock must retain its gradient; detaching it changes the training problem. The new second Taylor coefficient includes `.5 diag(P h) J_s(h) f(h)`, absent from the frozen conditional system. This is a different nonlinear field, not a correction of an erroneous original expansion. At unit horizon, midpoint also contains higher state-composition terms; it is not exactly the full second Taylor polynomial.

The source/protocol audit found no implementation or completed training of this exact contrast:

- [Stage5](protocol-clock-reuse.md) tied or reused controllers **between** two solves with different P1/P2 and a middle SiLU. Its [negative result](clock-reuse-readout.md) is relevant caution, but does not test an internal midpoint stage.
- [Stage7](protocol-nonlinear-drift.md) changed the field to `D0 P2 SiLU(P1 z)` with D0 frozen. Its [readout](nonlinear-drift-readout.md) confirms all trained arms were K1. The implemented K2/JVP test also freezes D0. Reopening that failed nonlinear-field family at higher order remains excluded.
- [Stage20](protocol-forced-gate.md) changed the scalar activation inside SwiGLU and prohibited its coupling/order rescues. None of that block, its lambda parameters, or its forced-response activation belongs in this question.

## What this can actually explain

For the original positive **scalar** clock, the exact autonomous trajectory satisfies `z(u)=exp(alpha(u)P)h`, where `alpha'=s(exp(alpha P)h)`. It follows the same linear orbit with different elapsed time. It creates no new orbit geometry. Vector clocks can change direction, but that observation is not a full-network expressivity or efficiency theorem.

The testable hypothesis is narrower: evaluating the **same** controller on the midpoint state may supply useful shared-parameter features that evaluating it only on h does not supply at this budget. The existing vector free-clock diagnostic already has essentially zero pointwise residual floor; this is not a claim that vector clocks lack pointwise control directions. It changes their shared functional dependence and optimization.

Neither an internal auxiliary-time ODE nor a better numerical approximation establishes compatibility with FM time, the complete velocity Jacobian, or the Gaussian-source covariance identity. A midpoint model beating cached K2 would support this architecture-level reevaluation choice on the tested tasks. It would not establish uniquely ODE-specific gains: the same computation is a tied two-stage residual network.

## Controls that make the screen interpretable

1. **Cached-clock K2 is indispensable.** Keep the same P, controller, middle SiLU, surrounding backbone, loss and horizon. This isolates reevaluating the clock at m. K1 alone conflates that change with a second projection stage.
2. **Pay for every evaluation.** For width d and vector-controller hidden width r, one field evaluation costs `d^2+2dr` dense MACs. Across the two solves, K1 costs `2d^2+4dr`, cached K2 `4d^2+4dr`, and midpoint `4d^2+8dr`. At d=64,r=32 these are 16,384, 24,576 and 32,768 per FFN. Nominal SwiGLU256 has actual hidden170 and 32,640 MACs: a close, not exact, inference-budget comparator. Count the common backbone and measure runtime separately. Longer training of ordinary SwiGLU is the distinct training-budget comparison; parameter sharing does not make repeated work free.
3. **Pair initial functions without claiming equal tangents.** Zero final controller weights/bias give constant D=cI, so cached K2 and midpoint initially agree. If a controller perturbation produces q(z), their first-order difference is `diag(Pm)[q(m)-q(h)]`. Last-layer controller weight features therefore differ; bias-only directions coincide. A zero final velocity head masks hidden-parameter tangents initially, so use a nonzero-head or local-FFN check. This constant-clock initialization also differs from legacy initialization: rerun the cached comparator rather than attributing all differences from historical K1 to state feedback.
4. **Keep a practical rejection rule.** An improvement only over cached K2, while losing to compute-matched SwiGLU, is not an architectural lead. Report all fixed task endpoints, field/Jacobian errors, sampling quality and actual costs. Failure ends this question without changing horizon, temperature, order, width or adding constraints.

## Direct literature scope

No new broad search is needed to establish midpoint mathematics. The Neural-ODE authors' [torchdiffeq Midpoint implementation](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/fixed_grid.py#L13-L19), inspected directly, computes the initial field, constructs the half-step state, and evaluates the field there. This supports the numerical interpretation only. Smoothness gives the usual small-step order; it does not guarantee accuracy, invertibility, stability or learning benefits for one learned unit-size step.

[Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning*, Section3](https://arxiv.org/html/2502.05171v2#S3) explicitly reapply a shared recurrent block to updated hidden states, including gated SiLU MLPs. It is directly relevant prior art for spending compute through state-dependent recurrence. Its language-model architecture, normalization and training differ from this two-stage FFN. It neither predicts our outcome nor makes gate reevaluation novel. The defensible contribution of the proposed screen is resolving an untested distinction in this project's original construction, not claiming a new solver or an established FM mechanism.
