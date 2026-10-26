# Saved-head output range is not the measured bottleneck

The original vector ODE's learned output envelope excludes too little of the target field to explain its current error. The conditional range floor is zero on Ring8 and only **0.0992%** of its field MSE on Product16. The stronger SwiGLU references have slightly larger floors. This weakens the specific explanation that a hard output-range limit drives the ODE/SwiGLU gap; it does not rule out other head-related optimization effects.

| Dataset | Arm | Actual field MSE | Envelope-distance floor | Floor / actual error | Targets outside |
|---|---|---:|---:|---:|---:|
| Ring8 | Vector parent | .016598458 | 0 | 0% | 0% |
| Ring8 | SwiGLU96 | .018921248 | .000001535 | .00811% | .0391% |
| Ring8 | SwiGLU128 | .014447538 | .000008560 | .05925% | .0391% |
| Product16 | Vector parent | .256487652 | .000254371 | .09917% | 1.6016% |
| Product16 | SwiGLU96 | .106571702 | .000441214 | .41401% | 3.0859% |
| Product16 | SwiGLU128 | .088920843 | .000371017 | .41724% | 2.6563% |

For the saved final LayerNorm and linear head, all outputs lie in `c+B*unit_ball`, with `c=W*beta+b` and `B=sqrt(width)*W*diag(gamma)*mean_zero_projector`. Both learned biases and the mean-zero restriction are included. The backbone need not realize the entire ellipsoid. Retraining can change its size and orientation, so the reported floor is conditional on these saved parameters, not an architecture-wide approximation bound.

All 30 existing panels are retained. Both the floor and actual error use mean squared velocity error per coordinate; the five-time summaries reproduce the native Stage17 values. Every saved prediction lies inside its envelope; the maximum squared ellipsoidal radius is .904866. No new model predictions, labels or training were generated.

The independent check constructs `B*B^T` algebraically, diagonalizes that Gram matrix, and solves the projection with safeguarded Newton iterations. It agrees with the original SVD/bisection implementation over every panel and all six aggregates, to maximum absolute difference `5.33e-15`. Explicit sphere cases return squared distances 0 and 9, and one 2D/16D parent panel is retained separately. The [independent record](independent-check.json), [all panels](per-panel.csv), [aggregates](aggregate.csv), and [source/input hashes](summary.json) preserve the evidence. The [registered protocol](source/protocol-head-envelope.md) defines the scope.
