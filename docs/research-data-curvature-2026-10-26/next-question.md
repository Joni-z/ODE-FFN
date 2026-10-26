# Next discriminating experiment: P coordinates or the nonlinear core?

Post-result research direction; not a launched trial and not an architecture nomination.

The current SwiGLU-target students have mostly saturated durations, locally modest clock-curvature contributions, and a fitting deficit that persists after globally fixing both clocks to 1. In the exact reduced space, roughly 68% of target complete curvature remains and the ODE fit still loses strongly. These observations leave two different explanations: learning through the polynomial effective matrices is difficult or restrictive, versus the fixed-width single-SiLU response family itself being inadequate under this budget. The eigenpair conditioning screen does not resolve the distinction or justify a generic spectral constraint.

A useful paired control starts from the same function:

```text
P coordinates: F(h) = Q(P2) SiLU(Q(P1) h), Q(P)=I+P+(1/2)P²
W coordinates: G(h) = W2 SiLU(W1 h), initialized with Wi=Q(Pi)
```

Both have two 64-by-64 learned matrices and no new activation, controller, bias or normalization. Compute the P arm through its effective Q matrices so the paired initial float32 functions can match bitwise, then verify the usual sequential polynomial form independently in float64. Native cold original P tensors supply initialization; no trained teacher weights enter a cold student.

Use two target families at all three positions. First, a fixed-clock ODE teacher constructed from the already frozen ODE teacher P matrices, evaluated through the same effective-map convention: it has an explicit zero-risk representer in both arms. Second, the existing unchanged SwiGLU128 local teacher, whose representability in either fixed-width core remains unknown. A known-representer contrast is necessary before attributing a P-versus-W gap to class absence. Keep a strong ordinary SwiGLU reference and every target/position visible. Existing identical SwiGLU-target runs may be reused only after validating actual initialization, index stream, target bytes and settings, never relabeled as new training.

Before launch, register all cells, exact target definitions, cold constructor pairing, optimizer/budget, measured dense work, function risks, P/effective-W trajectories, checkpoint/RNG/digest verification and interpretation. Preserve the fixed 5000-update budget and existing learning rate; do not combine a coordinate change with optimizer rescue or width expansion.

If W learns the known representable target substantially better, absence from the P class cannot explain that control's residual; finite-budget parameterization/optimization becomes a supported distinction for that function. If both learn the control but fail on SwiGLU targets, the simpler shared nonlinear core remains a plausible bottleneck. A gap only on the unknown-representability target leaves class restriction and optimization confounded. None of these diagnostic outcomes automatically supplies an ODE method; the W arm is an ordinary SiLU control.

This is distinct from the prior complete-network P-tangent cancellation measurement and the earlier generic expanded-SiLU FM reference. It isolates one local map in common coordinates with matched initial functions and an exact representer in both parameterizations. It does not reopen a stopped gate, bias, spectral, midpoint, forcing, augmentation or recurrence setting. Any later ODE design still needs ordinary FM, transfer and measured-cost evidence.
